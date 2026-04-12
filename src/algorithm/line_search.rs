use crate::manifolds::Manifold;
use crate::problem::Problem;
use crate::utils::traits::{Real, Vector};

/// Parameters for Armijo backtracking line search.
#[derive(Debug, Clone)]
pub struct BackTrackingParams<R>
where
    R: Real,
{
    /// Step size shrink factor, must lie in `(0, 1)`.
    tau: R,
    /// Initial trial step size.
    alpha0: R,
    /// Armijo sufficient decrease coefficient, must lie in `(0, 1)`.
    r: R,
    /// Minimum allowed step size before early success return.
    min_alpha: R,
    /// Maximum number of backtracking iterations.
    max_iters: usize,
}

impl<R> BackTrackingParams<R>
where
    R: Real,
{
    /// Create backtracking parameters with reduction `tau` and Armijo coefficient `r`.
    pub fn new(tau: R, r: R) -> Self {
        if tau <= R::zero() || tau >= R::one() {
            panic!("tau must be in (0, 1)");
        }
        if r <= R::zero() || r >= R::one() {
            panic!("r must be in (0, 1)");
        }
        Self {
            tau,
            alpha0: R::one(),
            r,
            min_alpha: R::zero(),
            max_iters: 64,
        }
    }

    /// Set initial step size.
    pub fn set_alpha0(mut self, alpha0: R) -> Self {
        if alpha0 <= R::zero() {
            panic!("alpha0 must be positive");
        }
        self.alpha0 = alpha0;
        self
    }

    /// Set maximum number of backtracking iterations.
    pub fn set_max_iters(mut self, max_iters: usize) -> Self {
        if max_iters == 0 {
            panic!("max_iters must be positive");
        }
        self.max_iters = max_iters;
        self
    }

    /// Set minimum allowed step size.
    pub fn set_min_alpha(mut self, min_alpha: R) -> Self {
        if min_alpha <= R::zero() {
            panic!("min_alpha must be positive");
        }
        self.min_alpha = min_alpha;
        self
    }
}

/// Line-search exit status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineSearchStatus {
    /// Armijo condition is satisfied.
    Success,
    /// Maximum iterations reached before success.
    MaxIters,
}

/// Backtracking line search on a manifold.
///
/// Returns `(alpha, next_point, status)` where:
/// - `alpha` is the accepted step size,
/// - `next_point` is `retraction(point, alpha * direction)`,
/// - `status` reports why the search stopped.
///
/// `covalue` is the directional derivative at `point` along `direction`.
pub fn back_tracking<R, M, F, G, H>(
    problem: &Problem<M, F, G, H>,
    point: &M::Point,
    value: R,
    direction: &M::TangentVector,
    covalue: R,
    params: &BackTrackingParams<R>,
) -> (R, M::Point, LineSearchStatus)
where
    R: Real,
    M: Manifold<Field = R>,
    F: Fn(&M::Point) -> R,
{
    let mut alpha = params.alpha0;
    let tau = params.tau;
    let r = params.r;

    let mut next_point = problem.retraction(point, &direction.ref_mul_num(alpha));

    for _ in 1..params.max_iters {
        let lhs = value - problem.value(&next_point);
        let rhs = r * alpha * covalue;
        if lhs >= rhs || alpha < params.min_alpha {
            return (alpha, next_point, LineSearchStatus::Success);
        }
        alpha *= tau;
        next_point = problem.retraction(point, &direction.ref_mul_num(alpha));
    }

    (alpha, next_point, LineSearchStatus::MaxIters)
}
