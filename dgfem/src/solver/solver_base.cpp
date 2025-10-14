#include "dgfem/core/space.hpp"
#include "dgfem/solver/dg_solver.hpp"

#include <filesystem>

#include <chrono>
#include <iostream>
#include <utility>

namespace dgfem {

DGSolverBase::DGSolverBase(std::shared_ptr<DGMesh> mesh, std::string solver_name)
    : mesh_(std::move(mesh)), solver_name_(std::move(solver_name)), output_directory_("output") {
    reset_telemetry();
    std::error_code ec;
    std::filesystem::create_directories(output_directory_, ec);
}

void DGSolverBase::set_solver_name(std::string name) {
    solver_name_ = std::move(name);
}

const std::string& DGSolverBase::solver_name() const noexcept {
    return solver_name_;
}

void DGSolverBase::set_verbose(bool verbose) noexcept {
    verbose_ = verbose;
}

bool DGSolverBase::verbose() const noexcept {
    return verbose_;
}

void DGSolverBase::set_output_directory(std::filesystem::path directory) {
    if (directory.empty()) {
        output_directory_.clear();
        return;
    }

    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    if (!ec) {
        output_directory_ = std::move(directory);
    } else if (verbose_) {
        log(Stage::Setup, "Failed to create output directory '" + directory.string() +
                              "'. Keeping previous path '" + output_directory_.string() + "'.");
    }
}

const std::filesystem::path& DGSolverBase::output_directory() const noexcept {
    return output_directory_;
}

void DGSolverBase::add_stage_observer(StageObserver observer) {
    observers_.push_back(std::move(observer));
}

void DGSolverBase::reset_telemetry() {
    telemetry_ = {};
    active_stage_flags_.fill(false);
    stage_start_time_.fill(Clock::time_point{});
}

const DGSolverBase::SolverTelemetry& DGSolverBase::telemetry() const noexcept {
    return telemetry_;
}

std::shared_ptr<DGSpace> DGSolverBase::space() const {
    if (!mesh_) {
        return nullptr;
    }
    return mesh_->get_dg_space();
}

int DGSolverBase::total_dofs() const {
    if (!mesh_) {
        return 0;
    }

    if (const auto solution = mesh_->get_solution(); solution) {
        return solution->get_total_dofs();
    }

    auto dg_space = mesh_->get_dg_space();
    if (!dg_space) {
        return 0;
    }

    const int n_basis = dg_space->get_basis()->get_n_basis();
    return mesh_->get_n_elements() * n_basis;
}

void DGSolverBase::begin_stage(Stage stage) const {
    const auto idx = static_cast<std::size_t>(stage);
    if (active_stage_flags_[idx]) {
        return;
    }
    active_stage_flags_[idx] = true;
    stage_start_time_[idx] = Clock::now();
}

void DGSolverBase::end_stage(Stage stage) const {
    const auto idx = static_cast<std::size_t>(stage);
    if (!active_stage_flags_[idx]) {
        return;
    }

    const auto end_time = Clock::now();
    const auto duration = std::chrono::duration<double>(end_time - stage_start_time_[idx]).count();

    active_stage_flags_[idx] = false;
    telemetry_.stages[idx].total_time += duration;
    telemetry_.stages[idx].count += 1;
    telemetry_.wall_time += duration;

    notify_observers(stage);
}

void DGSolverBase::increment_steps(int steps) noexcept {
    telemetry_.steps_completed += steps;
}

void DGSolverBase::increment_rhs_evaluations(int rhs) const noexcept {
    telemetry_.rhs_evaluations += rhs;
}

void DGSolverBase::increment_output_frames(int frames) noexcept {
    telemetry_.output_frames += frames;
}

void DGSolverBase::log(Stage stage, const std::string& message) const {
    if (!verbose_) {
        return;
    }
    std::cout << "[" << solver_name_ << "] [" << stage_to_string(stage) << "] " << message
              << std::endl;
}

void DGSolverBase::log(const std::string& message) const {
    if (!verbose_) {
        return;
    }
    std::cout << "[" << solver_name_ << "] " << message << std::endl;
}

std::filesystem::path DGSolverBase::make_output_path(const std::string& stem,
                                                     const std::string& extension) const {
    if (output_directory_.empty()) {
        return std::filesystem::path(stem + extension);
    }

    std::error_code ec;
    std::filesystem::create_directories(output_directory_, ec);

    if (!extension.empty() && extension.front() == '.') {
        return output_directory_ / (stem + extension);
    }
    if (!extension.empty()) {
        return output_directory_ / (stem + "." + extension);
    }
    return output_directory_ / stem;
}

std::string DGSolverBase::stage_to_string(Stage stage) {
    switch (stage) {
    case Stage::Setup:
        return "Setup";
    case Stage::Assembly:
        return "Assembly";
    case Stage::Solve:
        return "Solve";
    case Stage::Projection:
        return "Projection";
    case Stage::TimeStep:
        return "TimeStep";
    case Stage::Output:
        return "Output";
    case Stage::Postprocess:
        return "Postprocess";
    case Stage::Count:
    default:
        return "Unknown";
    }
}

void DGSolverBase::notify_observers(Stage stage) const {
    for (const auto& observer : observers_) {
        observer(stage, telemetry_);
    }
}

}  // namespace dgfem
