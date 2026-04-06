use derive_new::new;

use crate::algorithm::Status;
use crate::manifolds::Manifold;
use crate::problem::{FuncWithEGradEHess, Function, Problem};
use crate::utils::random_point::RandomOn;
use crate::utils::traits::{Real, Vector};

const DEFAULT_MIN_GRAD_NORM: f64 = 1e-6;
const DEFAULT_MIN_STEP_SIZE: f64 = 1e-12;
const DEFAULT_MAX_ITERATIONS: usize = 1000;
const DEFAULT_KAPPA: f64 = 0.1;
const DEFAULT_THETA: f64 = 1.0;
const DEFAULT_MAX_INNER_ITERATIONS: usize = 500;

/// Default Hessian-vector product callback type for RTR.
type HessianFunc<M> =
    fn(&<M as Manifold>::Point, &<M as Manifold>::TangentVector) -> <M as Manifold>::TangentVector;

// #[derive(Builder)]
/// Riemannian Trust-Region solver.
pub struct RTR<'a, 'b, R, M, F, H = HessianFunc<M>>
where
    R: Real,
    M: Manifold,
    F: Function<Point = M::Point, Field = R>,
    H: Fn(&M::Point, &M::TangentVector) -> M::TangentVector,
{
    problem: &'a mut Problem<'b, M, F>,
    min_grad_norm: R,
    min_step_size: R,
    max_iterations: usize,
    max_radius: R,
    threshold: R,
    kappa: R,
    theta: R,
    max_inner_iterations: usize,
    hessian_approx: Option<H>,
    verbose: u8,
}

#[derive(Debug, Clone, new)]
/// Result returned by `RTR::run`.
pub struct RTRResult<R, M>
where
    R: Real,
    M: Manifold,
{
    pub point: M::Point,
    pub final_value: R,
    pub final_grad_norm: R,
    pub iters: usize,
    pub status: Status,
}

impl<R: Real, M: Manifold> std::fmt::Display for RTRResult<R, M>
where
    M::Point: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RTRResult:\n")?;
        write!(f, "    final_value: {:.8e},\n", self.final_value)?;
        write!(f, "    final_grad_norm: {:.8e},\n", self.final_grad_norm)?;
        write!(f, "    iterations: {}.\n", self.iters)?;
        write!(f, "    status: {}.", self.status)
    }
}

impl<'a, 'b, R, M, F> RTR<'a, 'b, R, M, F>
where
    R: Real,
    M: Manifold,
    F: FuncWithEGradEHess<R, M::Point, M::AmbientPoint, M::TangentVector, M::AmbientPoint>,
{
    /// Create an RTR solver.
    ///
    /// `max_radius` is the upper bound for trust-region radius.
    /// `threshold` is the acceptance threshold for `rho`.
    pub fn new(problem: &'a mut Problem<'b, M, F>, max_radius: R, threshold: R) -> Self {
        Self {
            problem,
            min_grad_norm: R::from_f64(DEFAULT_MIN_GRAD_NORM),
            min_step_size: R::from_f64(DEFAULT_MIN_STEP_SIZE),
            max_iterations: DEFAULT_MAX_ITERATIONS,
            max_radius,
            threshold,
            kappa: R::from_f64(DEFAULT_KAPPA),
            theta: R::from_f64(DEFAULT_THETA),
            max_inner_iterations: DEFAULT_MAX_INNER_ITERATIONS,
            hessian_approx: None,
            verbose: 1,
        }
    }
}

impl<'a, 'b, R, M, F, H> RTR<'a, 'b, R, M, F, H>
where
    R: Real,
    M: Manifold<Field = R> + RandomOn,
    F: FuncWithEGradEHess<R, M::Point, M::AmbientPoint, M::TangentVector, M::AmbientPoint>,
    H: Fn(&M::Point, &M::TangentVector) -> M::TangentVector,
{
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

    /// Set maximum number of outer iterations.
    pub fn set_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set truncated-CG residual parameter `kappa`.
    pub fn set_kappa(mut self, kappa: R) -> Self {
        self.kappa = kappa;
        self
    }

    /// Set truncated-CG residual exponent `theta`.
    pub fn set_theta(mut self, theta: R) -> Self {
        self.theta = theta;
        self
    }

    /// Set maximum number of inner truncated-CG iterations.
    pub fn set_max_inner_iterations(mut self, max_inner_iterations: usize) -> Self {
        self.max_inner_iterations = max_inner_iterations;
        self
    }

    /// Set verbosity level (`0` disables logs).
    pub fn set_verbose(mut self, verbose: u8) -> Self {
        self.verbose = verbose;
        self
    }

    /// Replace exact Hessian-vector product with a user-provided approximation.
    pub fn set_hessian_approx<H2>(self, hess_approx: H2) -> RTR<'a, 'b, R, M, F, H2>
    where
        H2: Fn(&M::Point, &M::TangentVector) -> M::TangentVector,
    {
        RTR {
            problem: self.problem,
            min_grad_norm: self.min_grad_norm,
            min_step_size: self.min_step_size,
            max_iterations: self.max_iterations,
            max_radius: self.max_radius,
            threshold: self.threshold,
            kappa: self.kappa,
            theta: self.theta,
            max_inner_iterations: self.max_inner_iterations,
            hessian_approx: Some(hess_approx),
            verbose: self.verbose,
        }
    }

    #[inline]
    fn get_hessian(&self, point: &M::Point, tangent_vec: &M::TangentVector) -> M::TangentVector {
        if let Some(hess_approx) = &self.hessian_approx {
            hess_approx(point, tangent_vec)
        } else {
            self.problem.hessian(point, tangent_vec)
        }
    }

    // subproblem: m(s) = .5<s, Hs>_x - <b, s>_x, ||s|| <= radius, where b = -gradf_x.
    fn truncate_cg(
        &self,
        point: &M::Point,
        b: &M::TangentVector,
        radius: R,
    ) -> (M::TangentVector, R, Option<usize>) {
        let mut v = b.zeros_like();
        let mut r = b.clone();
        let mut p = r.clone();
        if self.problem.norm(point, &r) < R::epsilon() {
            return (v, R::zero(), Some(0));
        }

        let subproblem_func = |s| {
            let hs = self.get_hessian(point, s);
            R::half() * self.problem.inner(point, s, &hs) - self.problem.inner(point, b, s)
        };

        let b_norm = self.problem.norm(point, b);
        let r_bound = b_norm * R::min(self.kappa, b_norm.powf(self.theta));

        for iter in 1..self.max_inner_iterations {
            let hp = self.get_hessian(point, &p);
            let p_hp = self.problem.inner(point, &p, &hp);
            let alpha = self.problem.norm(point, &r).powi(2) / p_hp;
            let v_next = v.ref_add(p.ref_mul_num(alpha));

            if p_hp <= R::zero() || self.problem.norm(point, &v_next) >= radius {
                let inner_v_p = self.problem.inner(point, &v, &p);
                let t = -inner_v_p * R::two()
                    + R::sqrt(
                        inner_v_p.powi(2)
                            - R::from_f64(4.)
                                * self.problem.inner(point, &p, &p)
                                * (self.problem.inner(point, &v, &v) - radius.powi(2)),
                    );

                v = v + p * t;
                let subproblem_value = subproblem_func(&v);
                return (v, subproblem_value, Some(iter));
            }
            v = v_next;
            let r_next = r.ref_sub(hp.ref_mul_num(alpha));
            if self.problem.norm(point, &r) < r_bound {
                let subproblem_value = subproblem_func(&v);
                return (v, subproblem_value, Some(iter));
            }

            let beta =
                self.problem.norm(point, &r_next).powi(2) / self.problem.norm(point, &r).powi(2);
            p = r_next.ref_add(p * beta);
            r = r_next;
        }

        let subproblem_value = subproblem_func(&v);
        return (v, subproblem_value, None);
    }

    /// Run trust-region optimization from the problem's initial point.
    pub fn run(&mut self, mut radius: M::Field) -> RTRResult<R, M> {
        let mut current_point = self.problem.get_or_init_initial_point().clone();
        let mut current_value = self.problem.value(&current_point);
        let mut grad = self.problem.gradient(&current_point);
        let mut grad_norm = self.problem.norm(&current_point, &grad);

        if grad_norm < self.min_grad_norm {
            return RTRResult::new(
                current_point,
                current_value,
                grad_norm,
                0,
                Status::MinGradientNorm,
            );
        }

        for iter in 1..self.max_iterations {
            let (step, next_subproblem_value, inner_iters) =
                self.truncate_cg(&current_point, &-grad, radius);
            let next_point = self.problem.retraction(&current_point, &step);
            let next_value = self.problem.value(&next_point);

            grad = self.problem.gradient(&next_point);
            grad_norm = self.problem.norm(&next_point, &grad);

            if self.problem.norm(&current_point, &step) < self.min_step_size {
                return RTRResult::new(
                    next_point,
                    next_value,
                    grad_norm,
                    iter,
                    Status::MinStepSize,
                );
            }

            if grad_norm < self.min_grad_norm {
                return RTRResult::new(
                    next_point,
                    next_value,
                    grad_norm,
                    iter,
                    Status::MinGradientNorm,
                );
            }

            let rho = (current_value - next_value) / -next_subproblem_value;

            if rho > self.threshold {
                current_point = next_point;
                current_value = next_value;
            }

            if rho < R::from_f64(0.25) {
                radius = radius * R::from_f64(0.25);
            } else if rho > R::from_f64(0.75)
                || (self.problem.norm(&current_point, &step) - radius).abs() < R::epsilon()
            {
                radius = R::min(radius * R::two(), self.max_radius);
            }

            if self.verbose > 0 {
                println!(
                    "Iter: {}, Inner iters: {}, Cost: {:.8e}, Grad Norm: {:.8e}, Radius: {:.8e}, rho: {:.4}",
                    iter,
                    inner_iters.map_or("max".to_string(), |x| x.to_string()),
                    next_value,
                    grad_norm,
                    radius,
                    rho
                );
            }
        }

        RTRResult::new(
            current_point,
            current_value,
            grad_norm,
            self.max_iterations,
            Status::MaxIters,
        )
    }
}
