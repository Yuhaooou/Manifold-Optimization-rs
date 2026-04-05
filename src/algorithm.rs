//! Optimization algorithms and shared status types.

// use crate::manifolds::Manifold;
// use crate::problem::{Gradient, Problom};
// use crate::utils::Float;

pub mod line_search;
pub mod rgd;
pub mod rtr;

pub use line_search::{BackTrackingParams, back_tracking};
pub use rgd::{RGD, RGDResult};
pub use rtr::{RTR, RTRResult};

#[derive(Debug, Clone, PartialEq, Eq)]
/// Unified termination status for optimization algorithms.
pub enum Status {
    /// Stopping criterion: gradient norm is below threshold.
    MinGradientNorm,
    /// Stopping criterion: step size is below threshold.
    MinStepSize,
    /// Custom success reason.
    CustomSuccess(String),

    /// Iteration budget exhausted.
    MaxIters,
    /// Custom failure reason.
    CustomFailure(String),
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Status::MinGradientNorm => write!(f, "Minimum gradient norm reached"),
            Status::MinStepSize => write!(f, "Minimum step size reached"),
            Status::CustomSuccess(msg) => write!(f, "Success: {}", msg),
            Status::MaxIters => write!(f, "Maximum iterations reached"),
            Status::CustomFailure(msg) => write!(f, "Failure: {}", msg),
        }
    }
}

impl Status {
    /// Return `true` when the status denotes successful termination.
    pub fn is_success(&self) -> bool {
        matches!(
            self,
            Status::MinGradientNorm | Status::MinStepSize | Status::CustomSuccess(_)
        )
    }

    /// Return `true` when the status denotes failed termination.
    pub fn is_failure(&self) -> bool {
        matches!(self, Status::MaxIters | Status::CustomFailure(_))
    }
}

// pub trait Algorithm {
// }
