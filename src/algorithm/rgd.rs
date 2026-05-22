use crate::algorithm::Status;
use crate::algorithm::line_search::{BackTrackingParams, back_tracking};
use crate::function::FuncOne;
use crate::manifolds::Manifold;
use crate::problem::Problem;
use crate::utils::traits::{Real, Vector};

const DEFAULT_MIN_GRAD_NORM: f64 = 1e-8;
const DEFAULT_MIN_STEP_SIZE: f64 = 1e-12;
const DEFAULT_MAX_ITERATIONS: usize = 1000;

/// Riemannian Gradient Descent solver.
pub struct RGD<R, M, F>
where
    R: Real,
    M: Manifold,
    F: FuncOne<Manifold = M>,
{
    problem: Problem<M, F>,
    min_grad_norm: R,
    min_step_size: R,
    max_iterations: usize,
    back_tracking_params: BackTrackingParams<R>,
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
        write!(
            f,
            "    final_value: {:.8e},\n",
            self.final_value.to_f64().unwrap()
        )?;
        write!(
            f,
            "    grad_norm: {:.8e},\n",
            self.final_grad_norm.to_f64().unwrap()
        )?;
        write!(f, "    iterations: {}.\n", self.iters)?;
        write!(f, "    status: {}.", self.status)
    }
}

impl<R, M, F> RGD<R, M, F>
where
    R: Real,
    M: Manifold<Field = R>,
    F: FuncOne<Manifold = M>,
{
    /// Create an RGD solver with default stopping parameters.
    pub fn new(problem: Problem<M, F>, linesearch_params: BackTrackingParams<R>) -> Self {
        Self {
            problem,
            min_grad_norm: R::from_f64(DEFAULT_MIN_GRAD_NORM).unwrap(),
            min_step_size: R::from_f64(DEFAULT_MIN_STEP_SIZE).unwrap(),
            max_iterations: DEFAULT_MAX_ITERATIONS,
            back_tracking_params: linesearch_params,
            verbose: 1,
        }
    }

    pub fn problem_move(self) -> Problem<M, F> {
        self.problem
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
        let point = self.problem.get_initial_point().clone();

        self.problem.update_value_and_gradient(point);

        let mut grad_norm = self.problem.norm(self.problem.get_gradient());

        if grad_norm < self.min_grad_norm {
            return RGDResult {
                final_value: self.problem.get_value(),
                point: self.problem.return_point(),
                final_grad_norm: grad_norm,
                iters: 0,
                status: Status::MinGradientNorm,
            };
        }

        for iter in 1..=self.max_iterations {
            let (alpha, next_point, _) = back_tracking(
                &self.problem,
                self.problem.get_point(),
                self.problem.get_value(),
                &self.problem.get_gradient().ref_neg(),
                grad_norm.powi_(2),
                &self.back_tracking_params,
            );

            self.problem.update_value_and_gradient(next_point);

            grad_norm = self.problem.norm(self.problem.get_gradient());

            if alpha < self.min_step_size {
                return RGDResult {
                    final_value: self.problem.get_value(),
                    point: self.problem.return_point(),
                    final_grad_norm: grad_norm,
                    iters: iter,
                    status: Status::MinStepSize,
                };
            }

            if grad_norm < self.min_grad_norm {
                return RGDResult {
                    final_value: self.problem.get_value(),
                    point: self.problem.return_point(),
                    final_grad_norm: grad_norm,
                    iters: iter,
                    status: Status::MinGradientNorm,
                };
            }

            if self.verbose > 0 {
                println!(
                    "Iter: {}, Cost: {:.8e}, Grad Norm: {:.8e}, Step Size: {:.8e}",
                    iter,
                    self.problem.get_value().to_f64().unwrap(),
                    grad_norm.to_f64().unwrap(),
                    alpha.to_f64().unwrap()
                );
            }
        }

        RGDResult {
            final_value: self.problem.get_value(),
            point: self.problem.return_point(),
            final_grad_norm: grad_norm,
            iters: self.max_iterations,
            status: Status::MaxIters,
        }
    }
}
