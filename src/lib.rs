//! Manifold optimization primitives and algorithms.
//!
//! This crate provides:
//! - Core optimization problem traits in the `problem` module.
//! - Manifold definitions in the `manifolds` module.
//! - Solvers such as RGD/RTR in the `algorithm` module.
//! - Shared numeric/vector utilities in the `utils` module.

pub mod algorithm;
pub mod manifolds;
pub mod problem;
pub mod linalg;
pub mod utils;
