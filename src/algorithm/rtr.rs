use derive_new::new;

use crate::algorithm::Status;
use crate::manifolds::Manifold;
use crate::problem::Problem;
use crate::utils::traits::{Real, Vector};

const DEFAULT_MIN_GRAD_NORM: f64 = 1e-6;
const DEFAULT_MIN_STEP_SIZE: f64 = 1e-12;
const DEFAULT_MAX_ITERATIONS: usize = 1000;
const DEFAULT_KAPPA: f64 = 0.1;
const DEFAULT_THETA: f64 = 1.0;
const DEFAULT_MAX_INNER_ITERATIONS: usize = 500;

/// Riemannian Trust-Region solver.
pub struct RTR<'a, 'b, R, M, F, G, H>
where
    R: Real,
    M: Manifold,
    F: Fn(&M::Point) -> M::Field,
    G: Fn(&M::Point) -> M::TangentVector,
    H: Fn(&M::Point, &M::TangentVector) -> M::TangentVector,
{
    problem: &'a Problem<'b, M, F, G, H>,
    min_grad_norm: R,
    min_step_size: R,
    max_iterations: usize,
    max_radius: R,
    threshold: R,
    kappa: R,
    theta: R,
    max_inner_iterations: usize,
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
        write!(
            f,
            "    final_value: {:.8e},\n",
            self.final_value.to_f64().unwrap()
        )?;
        write!(
            f,
            "    final_grad_norm: {:.8e},\n",
            self.final_grad_norm.to_f64().unwrap()
        )?;
        write!(f, "    iterations: {}.\n", self.iters)?;
        write!(f, "    status: {}.", self.status)
    }
}

impl<'a, 'b, R, M, F, G, H> RTR<'a, 'b, R, M, F, G, H>
where
    R: Real,
    M: Manifold<Field = R>,
    F: Fn(&M::Point) -> M::Field,
    G: Fn(&M::Point) -> M::TangentVector,
    H: Fn(&M::Point, &M::TangentVector) -> M::TangentVector,
{
    pub fn new(problem: &'a mut Problem<'b, M, F, G, H>, max_radius: R, threshold: R) -> Self {
        RTR {
            problem,
            min_grad_norm: R::from_f64(DEFAULT_MIN_GRAD_NORM).unwrap(),
            min_step_size: R::from_f64(DEFAULT_MIN_STEP_SIZE).unwrap(),
            max_iterations: DEFAULT_MAX_ITERATIONS,
            max_radius,
            threshold,
            kappa: R::from_f64(DEFAULT_KAPPA).unwrap(),
            theta: R::from_f64(DEFAULT_THETA).unwrap(),
            max_inner_iterations: DEFAULT_MAX_INNER_ITERATIONS,
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
        if self.problem.norm(point, &r) == R::zero() {
            return (v, R::zero(), Some(0));
        }

        let subproblem_func = |s| {
            let hs = self.problem.hessian(point, s);
            R::half() * self.problem.inner(point, s, &hs) - self.problem.inner(point, b, s)
        };

        let b_norm = self.problem.norm(point, b);
        let r_bound = b_norm * R::min(self.kappa, b_norm.powf(self.theta));

        for iter in 1..=self.max_inner_iterations {
            let hp = self.problem.hessian(point, &p);
            let p_hp = self.problem.inner(point, &p, &hp);
            let alpha = self.problem.norm(point, &r).powi(2) / p_hp;
            let v_next = v.ref_add(p.ref_mul_num(alpha));

            if p_hp <= R::zero() || self.problem.norm(point, &v_next) >= radius {
                let inner_v_p = self.problem.inner(point, &v, &p);
                let norm_p_sq = self.problem.norm(point, &p).powi(2);
                let norm_v_sq = self.problem.norm(point, &v).powi(2);
                let tmp = norm_p_sq * (norm_v_sq - radius.powi(2)).muli(4);
                let t = (-inner_v_p + (inner_v_p.powi(2) - tmp).sqrt()) / norm_p_sq;

                v = v + p * t;
                let subproblem_value = subproblem_func(&v);
                return (v, subproblem_value, Some(iter));
            }
            v = v_next;
            let r_next = r.ref_sub(hp.ref_mul_num(alpha));
            if self.problem.norm(point, &r_next) < r_bound {
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
        let mut current_point = self.problem.get_initial_point().clone();
        let mut current_value = self.problem.function(&current_point);
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

        for iter in 1..=self.max_iterations {
            let (step, next_subproblem_value, inner_iters) =
                self.truncate_cg(&current_point, &-grad, radius);
            let next_point = self.problem.retraction(&current_point, &step);
            let next_value = self.problem.function(&next_point);

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

            if rho < R::from_f64(0.25).unwrap() {
                radius = radius * R::from_f64(0.25).unwrap();
            } else if rho > R::from_f64(0.75).unwrap()
                || (self.problem.norm(&current_point, &step) - radius).abs() == R::zero()
            {
                radius = R::min(radius.muli(2), self.max_radius);
            }

            if self.verbose > 0 {
                println!(
                    "Iter: {}, Inner iters: {}, Cost: {:.8e}, Grad Norm: {:.8e}, Radius: {:.8e}, rho: {:.4}",
                    iter,
                    inner_iters.map_or("max".to_string(), |x| x.to_string()),
                    current_value.to_f64().unwrap(),
                    grad_norm.to_f64().unwrap(),
                    radius.to_f64().unwrap(),
                    rho.to_f64().unwrap()
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
