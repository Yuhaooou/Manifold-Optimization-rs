use std::{any::Any, collections::HashMap, mem::replace};

use num_complex::ComplexFloat;

use crate::{
    function::{FuncOne, FuncZero},
    manifolds::manifold::*,
};

#[derive(Debug)]
pub struct Problem<M, F>
where
    M: Manifold,
    F: FuncZero<Manifold = M>,
{
    function: F,
    init_point: M::Point,
    caches: HashMap<String, Box<dyn Any>>,
}

impl<M, F> Problem<M, F>
where
    M: Manifold,
    F: FuncZero<Manifold = M>,
{
    /// Create a new problem with manifold, cost function and init point.
    pub fn new_with_init_point(function: F, init_point: M::Point) -> Self {
        Problem {
            function,
            init_point,
            caches: HashMap::new(),
        }
    }

    /// Get the configured initial point.
    pub fn get_initial_point(&self) -> &M::Point {
        &self.init_point
    }

    pub fn get_initial_point_mut(&mut self) -> &mut M::Point {
        &mut self.init_point
    }

    /// Set a new initial point, returning the old one.
    pub fn set_new_initial_point(&mut self, new_init: M::Point) -> M::Point {
        replace(&mut self.init_point, new_init)
    }

    pub fn manifold(&self) -> &M {
        self.function.manifold()
    }

    pub fn function(&self) -> &F {
        &self.function
    }

    pub fn function_mut(&mut self) -> &mut F {
        &mut self.function
    }

    pub fn get_cache<T: 'static>(&self, key: &str) -> Option<&T> {
        self.caches.get(key)?.downcast_ref::<T>()
    }

    pub fn get_cache_mut<T: 'static>(&mut self, key: &str) -> Option<&mut T> {
        self.caches.get_mut(key)?.downcast_mut::<T>()
    }

    pub fn insert_cache<T: 'static>(&mut self, key: String, value: T) {
        self.caches.insert(key, Box::new(value));
    }

    pub fn set_cache<T: 'static>(&mut self, key: String, value: T) {
        self.caches.insert(key, Box::new(value));
    }

    pub fn remove_cache(&mut self, key: &str) -> Option<Box<dyn Any>> {
        self.caches.remove(key)
    }

    pub fn clear_caches(&mut self) {
        self.caches.clear();
    }

    pub fn show_caches_keys(&self) {
        for key in self.caches.keys() {
            println!("{key}");
        }
    }
}

impl<M, F> Problem<M, F>
where
    M: Manifold + RandomPoint,
    F: FuncZero<Manifold = M>,
{
    pub fn new(function: F) -> Self {
        let init_point = function.manifold().random_point();
        Self::new_with_init_point(function, init_point)
    }

    pub fn new_with_rng<R: rand::Rng + ?Sized>(function: F, rng: &mut R) -> Self {
        let init_point = function.manifold().random_point_with_rng(rng);
        Self::new_with_init_point(function, init_point)
    }
}

impl<M, F> Problem<M, F>
where
    M: Manifold,
    F: FuncZero<Manifold = M>,
{
    /// Norm induced by manifold metric.
    pub fn norm(&self, v: &M::TangentVector) -> <M::Field as ComplexFloat>::Real {
        self.function.manifold().norm(self.function.get_point(), v)
    }

    /// Manifold inner product of tangent vectors.
    pub fn inner(
        &self,
        v1: &M::TangentVector,
        v2: &M::TangentVector,
    ) -> <M::Field as ComplexFloat>::Real {
        self.function
            .manifold()
            .inner(self.function.get_point(), v1, v2)
    }

    /// Retract a tangent vector back to the manifold.
    pub fn retraction(&self, v: &M::TangentVector) -> M::Point {
        self.function
            .manifold()
            .retraction(self.function.get_point(), v)
    }

    /// Project an ambient vector to tangent space.
    pub fn projection(&self, v: &M::AmbientPoint) -> M::TangentVector {
        self.function
            .manifold()
            .projection(self.function.get_point(), v)
    }

    pub fn compute_value(&self, x: &M::Point) -> M::Field {
        self.function.compute_value(x)
    }

    pub fn update_value(&mut self, x: M::Point) -> M::Field {
        self.function.update_value(x)
    }

    pub fn get_value(&self) -> M::Field {
        self.function.get_value()
    }

    pub fn get_point(&self) -> &M::Point {
        self.function.get_point()
    }

    pub fn return_point(&mut self) -> M::Point {
        self.function.return_point()
    }
}

impl<M, F> Problem<M, F>
where
    M: Manifold,
    F: FuncOne<Manifold = M>,
{
    pub fn update_value_and_gradient(&mut self, x: M::Point) -> M::Field {
        self.function.update_value_gradient(x)
    }

    pub fn get_gradient(&self) -> &M::TangentVector {
        self.function.get_gradient()
    }
}

impl<M, F> Problem<M, F>
where
    M: Manifold + Exp,
    F: FuncZero<Manifold = M>,
{
    pub fn exp(&self, v: &M::TangentVector) -> M::Point {
        self.function.manifold().exp(self.function.get_point(), v)
    }
}

impl<M, F> Problem<M, F>
where
    M: Manifold + Log,
    F: FuncZero<Manifold = M>,
{
    pub fn log(&self, y: &M::Point) -> M::TangentVector {
        self.function.manifold().log(self.function.get_point(), y)
    }
}

impl<M, F> Problem<M, F>
where
    M: Manifold + Transport,
    F: FuncZero<Manifold = M>,
{
    pub fn transport(&self, y: &M::Point, v: &M::TangentVector) -> M::TangentVector {
        self.function
            .manifold()
            .transport(self.function.get_point(), y, v)
    }
}
