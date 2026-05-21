use std::{any::Any, collections::HashMap, ptr::NonNull};

use num_complex::ComplexFloat;

use crate::manifolds::manifold::*;

/// Only cost function, for zero-order algorithms.
pub trait FuncZero {
    type Manifold: Manifold;

    fn manifold(&self) -> &Self::Manifold;

    fn function(
        &self,
        x: &<Self::Manifold as Manifold>::Point,
    ) -> <Self::Manifold as Manifold>::Field;
}

/// Cost function and gradient, for first-order algorithms.
pub trait FuncOne: FuncZero {
    fn gradient(
        &self,
        x: &<Self::Manifold as Manifold>::Point,
    ) -> <Self::Manifold as Manifold>::TangentVector;
}

/// Cost function, gradient and hessian, for second-order algorithms.
pub trait FuncTwo: FuncOne {
    fn hessian(
        &self,
        x: &<Self::Manifold as Manifold>::Point,
        v: &<Self::Manifold as Manifold>::TangentVector,
    ) -> <Self::Manifold as Manifold>::TangentVector;
}

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
        std::mem::replace(&mut self.init_point, new_init)
    }

    pub fn function(&self, x: &M::Point) -> M::Field {
        self.function.function(x)
    }

    /// Norm induced by manifold metric.
    pub fn norm(&self, x: &M::Point, v: &M::TangentVector) -> <M::Field as ComplexFloat>::Real {
        self.function.manifold().norm(x, v)
    }

    /// Manifold inner product of tangent vectors.
    pub fn inner(
        &self,
        x: &M::Point,
        v1: &M::TangentVector,
        v2: &M::TangentVector,
    ) -> <M::Field as ComplexFloat>::Real {
        self.function.manifold().inner(x, v1, v2)
    }

    /// Retract a tangent vector back to the manifold.
    pub fn retraction(&self, x: &M::Point, v: &M::TangentVector) -> M::Point {
        self.function.manifold().retraction(x, v)
    }

    /// Project an ambient vector to tangent space.
    pub fn projection(&self, x: &M::Point, v: &M::AmbientPoint) -> M::TangentVector {
        self.function.manifold().projection(x, v)
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
    F: FuncOne<Manifold = M>,
{
    pub fn gradient(&self, x: &M::Point) -> M::TangentVector {
        self.function.gradient(x)
    }
}

impl<M, F> Problem<M, F>
where
    M: Manifold,
    F: FuncTwo<Manifold = M>,
{
    pub fn hessian(&self, x: &M::Point, v: &M::TangentVector) -> M::TangentVector {
        self.function.hessian(x, v)
    }
}

pub struct FuncOnly<M, F>
where
    M: Manifold,
    F: Fn(&M::Point) -> M::Field,
{
    function: F,
    manifold: M,
}

impl<M, F> FuncOnly<M, F>
where
    M: Manifold,
    F: Fn(&M::Point) -> M::Field,
{
    pub fn new(manifold: M, function: F) -> Self {
        FuncOnly { function, manifold }
    }
}

impl<M, F> FuncZero for FuncOnly<M, F>
where
    M: Manifold,
    F: Fn(&M::Point) -> M::Field,
{
    type Manifold = M;

    fn manifold(&self) -> &Self::Manifold {
        &self.manifold
    }

    fn function(&self, x: &M::Point) -> M::Field {
        (self.function)(x)
    }
}

pub struct FuncGrad<M, F>
where
    M: Manifold,
    F: Fn(&M::Point, bool, bool) -> (Option<M::Field>, Option<M::TangentVector>),
{
    manifold: M,
    fun_with_grad: F,
    point_ptr: Option<NonNull<M::Point>>,
    current_value: Option<M::Field>,
    current_grad: Option<M::TangentVector>,
}

impl<M, F> FuncGrad<M, F>
where
    M: Manifold,
    F: Fn(&M::Point, bool, bool) -> (Option<M::Field>, Option<M::TangentVector>),
{
    pub fn new(manifold: M, fun_with_grad: F) -> Self {
        FuncGrad {
            manifold,
            fun_with_grad,
            point_ptr: None,
            current_value: None,
            current_grad: None,
        }
    }

    pub fn new_with_fun_grad<Fun, Grad>(manifold: M, 
        function: impl Fn(&M::Point) -> M::Field,
        gradient: impl Fn(&M::Point) -> M::TangentVector,
    ) -> FuncGrad<M, impl Fn(&M::Point, bool, bool) -> (Option<M::Field>, Option<M::TangentVector>)>
    {
        let fun_with_grad = move |x: &M::Point, compute_cost: bool, compute_grad: bool| {
            let value = if compute_cost {
                Some(function(x))
            } else {
                None
            };
            let grad = if compute_grad {
                Some(gradient(x))
            } else {
                None
            };
            (value, grad)
        };

        FuncGrad {
            manifold,
            fun_with_grad,
            point_ptr: None,
            current_value: None,
            current_grad: None,
        }
    }

    pub fn eval(
        &mut self,
        x: &M::Point,
        compute_cost: bool,
        compute_grad: bool,
    ) -> Option<M::Field> {
        self.point_ptr = Some(NonNull::from_ref(x));
        let (value, grad) = (self.fun_with_grad)(x, compute_cost, compute_grad);
        self.current_value = value;
        self.current_grad = grad;
        value
    }

    pub fn check_ptr(&self, x: &M::Point) -> bool {
        if let Some(ptr) = self.point_ptr {
            ptr == NonNull::from_ref(x)
        } else {
            false
        }
    }
}

impl<M, F> FuncGrad<M, F>
where
    M: Manifold + EGradToRGrad,
    F: Fn(&M::Point, bool, bool) -> (Option<M::Field>, Option<M::TangentVector>),
{
    pub fn new_with_fun_egrad(manifold: M, 
        function: impl Fn(&M::Point) -> M::Field,
        egradient: impl Fn(&M::Point) -> M::AmbientPoint,
    ) -> FuncGrad<M, impl Fn(&M::Point, bool, bool) -> (Option<M::Field>, Option<M::TangentVector>)>
    {
        let manifold_ = manifold.clone();
        let fun_with_grad = move |x: &M::Point, compute_cost: bool, compute_grad: bool| {
            let value = if compute_cost {
                Some(function(x))
            } else {
                None
            };
            let grad = if compute_grad {
                let egrad = egradient(x);
                Some(manifold_.egrad_to_rgrad(x, &egrad))
            } else {
                None
            };
            (value, grad)
        };

        FuncGrad {
            manifold,
            fun_with_grad,
            point_ptr: None,
            current_value: None,
            current_grad: None,
        }
    }
}

impl<M, F> FuncZero for FuncGrad<M, F>
where
    M: Manifold,
    F: Fn(&M::Point, bool, bool) -> (Option<M::Field>, Option<M::TangentVector>),
{
    type Manifold = M;

    fn manifold(&self) -> &Self::Manifold {
        &self.manifold
    }

    fn function(&self, x: &M::Point) -> M::Field {
        if let Some(ptr) = self.point_ptr
            && ptr == NonNull::from_ref(x)
            && self.current_value.is_some()
        {
            self.current_value.unwrap()
        } else {
            (self.fun_with_grad)(x, true, false).0.unwrap()
        }
    }
}

impl<M, F> FuncOne for FuncGrad<M, F>
where
    M: Manifold,
    F: Fn(&M::Point, bool, bool) -> (Option<M::Field>, Option<M::TangentVector>),
{
    fn gradient(
        &self,
        x: &<Self::Manifold as Manifold>::Point,
    ) -> <Self::Manifold as Manifold>::TangentVector {
        (self.fun_with_grad)(x, false, true).1.unwrap()
    }
}

pub struct FuncGradHess<M, F>
where
    M: Manifold,
    F: Fn(
        &M::Point,
        Option<&M::TangentVector>,
        bool,
        bool,
        bool,
    ) -> (
        Option<M::Field>,
        Option<M::TangentVector>,
        Option<M::TangentVector>,
    ),
{
    manifold: M,
    fun_with_grad_hess: F,
    point_ptr: Option<NonNull<M::Point>>,
    tangent_ptr: Option<NonNull<M::TangentVector>>,
    current_value: Option<M::Field>,
    current_grad: Option<M::TangentVector>,
    current_hess: Option<M::TangentVector>,
}

impl<M, F> FuncGradHess<M, F>
where
    M: Manifold,
    F: Fn(
        &M::Point,
        Option<&M::TangentVector>,
        bool,
        bool,
        bool,
    ) -> (
        Option<M::Field>,
        Option<M::TangentVector>,
        Option<M::TangentVector>,
    ),
{
    pub fn new(manifold: M, fun_with_grad_hess: F) -> Self {
        FuncGradHess {
            manifold,
            fun_with_grad_hess,
            point_ptr: None,
            tangent_ptr: None,
            current_value: None,
            current_grad: None,
            current_hess: None,
        }
    }

    pub fn new_with_fun_grad_hess<Fun, Grad, Hess>(manifold: M, 
        function: impl Fn(&M::Point) -> M::Field,
        gradient: impl Fn(&M::Point) -> M::TangentVector,
        hessian: impl Fn(&M::Point, &M::TangentVector) -> M::TangentVector,
    ) -> FuncGradHess<
        M,
        impl Fn(
            &M::Point,
            Option<&M::TangentVector>,
            bool,
            bool,
            bool,
        ) -> (
            Option<M::Field>,
            Option<M::TangentVector>,
            Option<M::TangentVector>,
        ),
    > {
        let fun_with_grad_hess = move |x: &M::Point,
                                       v: Option<&M::TangentVector>,
                                       compute_cost: bool,
                                       compute_grad: bool,
                                       compute_hess: bool| {
            let value = if compute_cost {
                Some(function(x))
            } else {
                None
            };
            let grad = if compute_grad {
                Some(gradient(x))
            } else {
                None
            };
            let hess = if compute_hess {
                Some(hessian(x, v.unwrap()))
            } else {
                None
            };
            (value, grad, hess)
        };

        FuncGradHess {
            manifold,
            fun_with_grad_hess,
            point_ptr: None,
            tangent_ptr: None,
            current_value: None,
            current_grad: None,
            current_hess: None,
        }
    }
}

impl<M, F> FuncZero for FuncGradHess<M, F>
where
    M: Manifold,
    F: Fn(
        &M::Point,
        Option<&M::TangentVector>,
        bool,
        bool,
        bool,
    ) -> (
        Option<M::Field>,
        Option<M::TangentVector>,
        Option<M::TangentVector>,
    ),
{
    type Manifold = M;

    fn manifold(&self) -> &Self::Manifold {
        &self.manifold
    }

    fn function(&self, x: &M::Point) -> M::Field {
        if let Some(ptr) = self.point_ptr
            && ptr == NonNull::from_ref(x)
            && self.current_value.is_some()
        {
            self.current_value.unwrap()
        } else {
            (self.fun_with_grad_hess)(x, None, true, false, false)
                .0
                .unwrap()
        }
    }
}

impl<M, F> FuncOne for FuncGradHess<M, F>
where
    M: Manifold,
    F: Fn(
        &M::Point,
        Option<&M::TangentVector>,
        bool,
        bool,
        bool,
    ) -> (
        Option<M::Field>,
        Option<M::TangentVector>,
        Option<M::TangentVector>,
    ),
{
    fn gradient(&self, x: &M::Point) -> M::TangentVector {
        (self.fun_with_grad_hess)(x, None, false, true, false)
            .1
            .unwrap()
    }
}

impl<M, F> FuncTwo for FuncGradHess<M, F>
where
    M: Manifold,
    F: Fn(
        &M::Point,
        Option<&M::TangentVector>,
        bool,
        bool,
        bool,
    ) -> (
        Option<M::Field>,
        Option<M::TangentVector>,
        Option<M::TangentVector>,
    ),
{
    fn hessian(&self, x: &M::Point, v: &M::TangentVector) -> M::TangentVector {
        (self.fun_with_grad_hess)(x, Some(v), false, false, true)
            .2
            .unwrap()
    }
}

impl<M, F> Problem<M, F>
where
    M: Manifold + Exp,
    F: FuncZero<Manifold = M>,
{
    pub fn exp(&self, x: &M::Point, v: &M::TangentVector) -> M::Point {
        self.function.manifold().exp(x, v)
    }
}

impl<M, F> Problem<M, F>
where
    M: Manifold + Log,
    F: FuncZero<Manifold = M>,
{
    pub fn log(&self, x: &M::Point, y: &M::Point) -> M::TangentVector {
        self.function.manifold().log(x, y)
    }
}

impl<M, F> Problem<M, F>
where
    M: Manifold + Transport,
    F: FuncZero<Manifold = M>,
{
    pub fn transport(&self, x: &M::Point, y: &M::Point, v: &M::TangentVector) -> M::TangentVector {
        self.function.manifold().transport(x, y, v)
    }
}
