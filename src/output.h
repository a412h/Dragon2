#ifndef OUTPUT_H
#define OUTPUT_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/mpi.h>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <filesystem>

#include "offline_data.h"

using namespace dealii;

constexpr bool enable_schlieren = true;

template <int dim, typename Number = double>
class VTUOutput {
public:
    VTUOutput(const DoFHandler<dim>& dof_handler,
              const std::string& basename,
              const OfflineData<dim, Number>& offline_data)
        : dof_handler(dof_handler), basename(basename), offline_data(offline_data) {
        auto parent = std::filesystem::path(basename).parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);
    }

    void write(const std::vector<std::array<Number, dim+2>>& U,
               unsigned int cycle,
               Number time) {

        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);

        const unsigned int n_dofs = dof_handler.n_dofs();

        Vector<Number> rho(n_dofs);
        std::vector<Vector<Number>> momentum(dim);
        for (int d = 0; d < dim; ++d) {
            momentum[d].reinit(n_dofs);
        }
        Vector<Number> E(n_dofs);
        Vector<Number> schlieren_rho(n_dofs);

        #pragma omp parallel for
        for (unsigned int i = 0; i < n_dofs; ++i) {
            rho[i] = U[i][0];
            for (int d = 0; d < dim; ++d) {
                momentum[d][i] = U[i][1 + d];
            }
            E[i] = U[i][dim + 1];
        }

        if constexpr (enable_schlieren) {
            compute_schlieren(rho, schlieren_rho);
        }

        data_out.add_data_vector(rho, "rho", DataOut<dim>::type_dof_data);

        if constexpr (dim == 1) {
            data_out.add_data_vector(momentum[0], "m_1", DataOut<dim>::type_dof_data);
        } else if constexpr (dim == 2) {
            data_out.add_data_vector(momentum[0], "m_1", DataOut<dim>::type_dof_data);
            data_out.add_data_vector(momentum[1], "m_2", DataOut<dim>::type_dof_data);
        } else if constexpr (dim == 3) {
            data_out.add_data_vector(momentum[0], "m_1", DataOut<dim>::type_dof_data);
            data_out.add_data_vector(momentum[1], "m_2", DataOut<dim>::type_dof_data);
            data_out.add_data_vector(momentum[2], "m_3", DataOut<dim>::type_dof_data);
        }

        data_out.add_data_vector(E, "E", DataOut<dim>::type_dof_data);
        if constexpr (enable_schlieren) {
            data_out.add_data_vector(schlieren_rho, "schlieren_rho", DataOut<dim>::type_dof_data);
        }

        const Triangulation<dim>& triangulation = dof_handler.get_triangulation();
        Vector<float> manifold_ids(triangulation.n_active_cells());
        unsigned int cell_index = 0;
        for (const auto& cell : dof_handler.active_cell_iterators()) {
            manifold_ids[cell_index] = -1.0f;
            for (const auto& face : cell->face_iterators()) {
                if (face->at_boundary()) {
                    manifold_ids[cell_index] = static_cast<float>(face->manifold_id());
                    break;
                }
            }
            ++cell_index;
        }
        data_out.add_data_vector(manifold_ids, "manifold_id", DataOut<dim>::type_cell_data);

        data_out.build_patches();

        std::ostringstream filename;
        filename << basename << "_" << std::setfill('0') << std::setw(6)
                 << cycle << ".vtu";

        std::ofstream output(filename.str());
        data_out.write_vtu(output);
        output.close();

        std::cout << "Output written: " << filename.str()
                  << " at t=" << std::fixed << std::setprecision(4) << time
                  << std::endl;
    }

private:
    const DoFHandler<dim>& dof_handler;
    const std::string basename;
    const OfflineData<dim, Number>& offline_data;

    static constexpr Number schlieren_beta = Number(10.0);

    void compute_schlieren(const Vector<Number>& rho, Vector<Number>& schlieren) {
        const unsigned int n_dofs = dof_handler.n_dofs();
        const auto& sparsity = offline_data.sparsity;
        const auto& c_ij = offline_data.c_ij;
        const auto& m_inv = offline_data.lumped_mass_matrix_inverse;

        #pragma omp parallel for
        for (unsigned int i = 0; i < n_dofs; ++i) {
            if (sparsity[i].size() == 1) {
                schlieren[i] = Number(0);
                continue;
            }

            std::array<Number, dim> grad_i;
            grad_i.fill(Number(0));

            for (size_t col_idx = 0; col_idx < sparsity[i].size(); ++col_idx) {
                const unsigned int j = sparsity[i][col_idx];
                const Number rho_j = rho[j];
                for (unsigned int d = 0; d < dim; ++d) {
                    grad_i[d] -= c_ij[i][col_idx][d] * rho_j;
                }
            }

            Number grad_mag_sq = Number(0);
            for (unsigned int d = 0; d < dim; ++d) {
                grad_i[d] *= m_inv[i];
                grad_mag_sq += grad_i[d] * grad_i[d];
            }

            schlieren[i] = std::sqrt(grad_mag_sq);
        }

        Number q_min = std::numeric_limits<Number>::max();
        Number q_max = Number(0);

        for (unsigned int i = 0; i < n_dofs; ++i) {
            const Number q = schlieren[i];
            q_min = std::min(q_min, q);
            q_max = std::max(q_max, q);
        }

        constexpr Number eps = std::numeric_limits<Number>::epsilon();
        constexpr Number floor = Number(1.0e-10) > eps ? Number(1.0e-10) : eps;
        const Number range = std::max(q_max - q_min, eps);

        #pragma omp parallel for
        for (unsigned int i = 0; i < n_dofs; ++i) {
            const Number q = schlieren[i];
            const Number ratio = std::max(Number(0), q - q_min - floor) / range;
            schlieren[i] = Number(1) - std::exp(-schlieren_beta * ratio);
        }
    }
};

template<int dim>
class AsyncVTUWriter {
private:
    std::thread writer_thread;
    std::queue<std::tuple<std::vector<std::array<double, dim+2>>, unsigned int, double>> write_queue;
    std::mutex queue_mutex;
    std::condition_variable cv_queue;
    std::condition_variable cv_empty;
    bool stop_flag = false;
    VTUOutput<dim>* output_handler;

public:
    AsyncVTUWriter(VTUOutput<dim>* handler) : output_handler(handler) {
        writer_thread = std::thread(&AsyncVTUWriter::writer_loop, this);
    }

    ~AsyncVTUWriter() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            stop_flag = true;
        }
        cv_queue.notify_all();

        if (writer_thread.joinable()) {
            writer_thread.join();
        }
    }

    void enqueue_write(std::vector<std::array<double, dim+2>>&& data,
                      unsigned int cycle, double time) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            write_queue.push({std::move(data), cycle, time});
        }
        cv_queue.notify_one();
    }

    void wait_for_completion() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        cv_empty.wait(lock, [this] { return write_queue.empty(); });
    }

private:
    void writer_loop() {
        while (true) {
            std::unique_lock<std::mutex> lock(queue_mutex);

            cv_queue.wait(lock, [this] { return !write_queue.empty() || stop_flag; });

            if (stop_flag && write_queue.empty()) {
                break;
            }

            while (!write_queue.empty()) {
                auto [data, cycle, time] = std::move(write_queue.front());
                write_queue.pop();

                if (write_queue.empty()) {
                    cv_empty.notify_all();
                }

                lock.unlock();

                output_handler->write(data, cycle, time);

                lock.lock();
            }
        }
    }
};

#endif