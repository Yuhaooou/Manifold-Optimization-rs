use crate::algorithm::Status;
use crate::algorithm::line_search::{BackTrackingParams, back_tracking};
use crate::manifolds::Manifold;
use crate::problem::{EGradient, Function, Problem};
use crate::utils::random_point::RandomOn;
use crate::utils::traits::Real;

const DEFAULT_MIN_GRAD_NORM: f64 = 1e-8;
const DEFAULT_MIN_STEP_SIZE: f64 = 1e-12;
const DEFAULT_MAX_ITERATIONS: usize = 1000;

/// Riemannian Gradient Descent solver.
pub struct RGD<'a, 'b, R, M, F>
where
    R: Real,
    M: Manifold,
    F: Function<Point = M::Point>,
{
    problem: &'a mut Problem<'b, M, F>,
    min_grad_norm: R,
    min_step_size: R,
    max_iterations: usize,
    back_tracking_params: &'a BackTrackingParams<R>,
    verbose: u8,
}

#[derive(Debug, Clone)]
/// Result returned by `RGD::run`.
pub struct RGDResult<R, M: Manifold>
where
    R: Real,
{
    pub point: M::Point,
    pub final_value: R,
    pub final_grad_norm: R,
    pub iters: usize,
    pub status: Status,
}

impl<R: Real, M: Manifold> std::fmt::Display for RGDResult<R, M>
where
    M::Point: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RGDResult:\n")?;
        write!(f, "    final_value: {},\n", self.final_value)?;
        write!(f, "    grad_norm: {},\n", self.final_grad_norm)?;
        write!(f, "    iterations: {}.", self.iters)?;
        write!(f, "    status: {}.", self.status)
    }
}

impl<'a, 'b, R, M, F> RGD<'a, 'b, R, M, F>
where
    R: Real,
    M: Manifold<Field = R> + RandomOn,
    F: Function<Point = M::Point, Field = R>
        + EGradient<Point = M::Point, Gradient = M::AmbientPoint>,
{
    /// Create an RGD solver with default stopping parameters.
    pub fn new(
        problem: &'a mut Problem<'b, M, F>,
        linesearch_params: &'a BackTrackingParams<R>,
    ) -> Self {
        Self {
            problem,
            min_grad_norm: R::from_f64(DEFAULT_MIN_GRAD_NORM),
            min_step_size: R::from_f64(DEFAULT_MIN_STEP_SIZE),
            max_iterations: DEFAULT_MAX_ITERATIONS,
            back_tracking_params: linesearch_params,
            verbose: 1,
        }
    }

    /// Set minimum gradient norm stopping threshold.
    pub fn set_min_grad_norm(mut self, min_grad_norm: R) -> Self {
        self.min_grad_norm = min_grad_norm;
        self
    }

    /// Set minimum accepted step size threshold.
    pub fn set_min_step_size(mut self, min_step_size: R) -> Self {
        self.min_step_size = min_step_size;
        self
    }

    /// Set maximum number of iterations.
    pub fn set_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set verbosity level (`0` disables logs).
    pub fn set_verbose(mut self, verbose: u8) -> Self {
        self.verbose = verbose;
        self
    }

    /// Run optimization until one stopping criterion is met.
    pub fn run(&mut self) -> RGDResult<R, M> {
        let mut current_point = self.problem.get_or_init_initial_point().clone();
        let mut current_value = self.problem.value(&current_point);
        let mut grad = self.problem.gradient(&current_point);
        let mut grad_norm = self.problem.norm(&current_point, &grad);

        if grad_norm < self.min_grad_norm {
            return RGDResult {
                final_value: self.problem.value(&current_point),
                point: current_point,
                final_grad_norm: grad_norm,
                iters: 0,
                status: Status::MinGradientNorm,
            };
        }

        for iter in 1..self.max_iterations {
            let (alpha, next_point, _) = back_tracking(
                &self.problem,
                &current_point,
                current_value,
                &-grad,
                grad_norm.powi(2),
                &self.back_tracking_params,
            );

            grad = self.problem.gradient(&next_point);
            grad_norm = self.problem.norm(&next_point, &grad);
            let next_value = self.problem.value(&next_point);

            if alpha < self.min_step_size {
                return RGDResult {
                    final_value: next_value,
                    point: next_point,
                    final_grad_norm: grad_norm,
                    iters: iter,
                    status: Status::MinStepSize,
                };
            }

            if grad_norm < self.min_grad_norm {
                return RGDResult {
                    final_value: next_value,
                    point: next_point,
                    final_grad_norm: grad_norm,
                    iters: iter,
                    status: Status::MinGradientNorm,
                };
            }

            if self.verbose > 0 {
                println!(
                    "Iter: {}, Cost: {:.8e}, Grad Norm: {:.8e}, Step Size: {:.8e}",
                    iter,
                    next_value.to_f64(),
                    grad_norm.to_f64(),
                    alpha.to_f64()
                );
            }

            current_point = next_point;
            current_value = next_value;
        }

        RGDResult {
            point: current_point,
            final_value: current_value,
            final_grad_norm: grad_norm,
            iters: self.max_iterations,
            status: Status::MaxIters,
        }
    }
}
