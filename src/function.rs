use std::mem::replace;

use crate::manifolds::{EGradToRGrad, EHessToRHess, Manifold};

/// Only cost function, for zero-order algorithms.
pub trait FuncZero {
    type Manifold: Manifold;

    fn manifold(&self) -> &Self::Manifold;

    fn compute_value(
        &self,
        point: &<Self::Manifold as Manifold>::Point,
    ) -> <Self::Manifold as Manifold>::Field;

    fn update_value(
        &mut self,
        point: <Self::Manifold as Manifold>::Point,
    ) -> <Self::Manifold as Manifold>::Field;

    fn get_value(&self) -> <Self::Manifold as Manifold>::Field;

    fn get_point(&self) -> &<Self::Manifold as Manifold>::Point;

    fn return_point(&mut self) -> <Self::Manifold as Manifold>::Point {
        self.replace_value_point(None, None)
    }

    fn replace_value_point(
        &mut self,
        value: Option<<Self::Manifold as Manifold>::Field>,
        point: Option<<Self::Manifold as Manifold>::Point>,
    ) -> <Self::Manifold as Manifold>::Point;
}

/// Cost function and gradient, for first-order algorithms.
pub trait FuncOne: FuncZero {
    fn compute_value_gradient(
        &self,
        point: &<Self::Manifold as Manifold>::Point,
    ) -> (
        <Self::Manifold as Manifold>::Field,
        <Self::Manifold as Manifold>::TangentVector,
    );

    fn update_value_gradient(
        &mut self,
        point: <Self::Manifold as Manifold>::Point,
    ) -> <Self::Manifold as Manifold>::Field;

    fn get_gradient(&self) -> &<Self::Manifold as Manifold>::TangentVector;

    fn return_gradient(&mut self) -> <Self::Manifold as Manifold>::TangentVector;

    /// Directly compute gradient. May be less efficient if the struct already has the same point, but no guarantee.
    fn directly_get_gradient(
        &self,
        point: &<Self::Manifold as Manifold>::Point,
    ) -> <Self::Manifold as Manifold>::TangentVector;

    fn return_point_gradient(
        &mut self,
    ) -> (
        <Self::Manifold as Manifold>::Point,
        <Self::Manifold as Manifold>::TangentVector,
    ) {
        self.replace_value_point_gradient(None, None, None)
    }

    fn replace_value_point_gradient(
        &mut self,
        value: Option<<Self::Manifold as Manifold>::Field>,
        point: Option<<Self::Manifold as Manifold>::Point>,
        grad: Option<<Self::Manifold as Manifold>::TangentVector>,
    ) -> (
        <Self::Manifold as Manifold>::Point,
        <Self::Manifold as Manifold>::TangentVector,
    );
}

/// Cost function, gradient and hessian, for second-order algorithms.
pub trait FuncTwo: FuncOne {
    fn compute_value_gradient_hessian(
        &mut self,
        x: &<Self::Manifold as Manifold>::Point,
        v: &<Self::Manifold as Manifold>::TangentVector,
    ) -> (
        <Self::Manifold as Manifold>::Field,
        <Self::Manifold as Manifold>::TangentVector,
        <Self::Manifold as Manifold>::TangentVector,
    );

    fn update_value_gradient_hessian(
        &mut self,
        x: <Self::Manifold as Manifold>::Point,
        v: <Self::Manifold as Manifold>::TangentVector,
    ) -> <Self::Manifold as Manifold>::Field;

    fn get_hessian(&self) -> &<Self::Manifold as Manifold>::TangentVector;

    fn return_hessian(&mut self) -> <Self::Manifold as Manifold>::TangentVector;

    fn directly_get_hessian(
        &self,
        x: &<Self::Manifold as Manifold>::Point,
        v: &<Self::Manifold as Manifold>::TangentVector,
    ) -> <Self::Manifold as Manifold>::TangentVector;
}

pub struct FuncOnly<M, F>
where
    M: Manifold,
    F: Fn(&M::Point) -> M::Field,
{
    manifold: M,
    function: F,
    current_point: Option<M::Point>,
    current_value: Option<M::Field>,
}

impl<M, F> FuncOnly<M, F>
where
    M: Manifold,
    F: Fn(&M::Point) -> M::Field,
{
    pub fn new(manifold: M, function: F) -> Self {
        FuncOnly {
            function,
            manifold,
            current_point: None,
            current_value: None,
        }
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

    fn compute_value(&self, x: &M::Point) -> M::Field {
        (self.function)(x)
    }

    fn update_value(&mut self, x: M::Point) -> M::Field {
        let value = self.compute_value(&x);
        self.current_value = Some(value);
        self.current_point = Some(x);
        value
    }

    fn get_value(&self) -> M::Field {
        self.current_value.expect("Update first")
    }

    fn get_point(&self) -> &M::Point {
        self.current_point.as_ref().expect("Point not contained")
    }

    fn replace_value_point(
        &mut self,
        value: Option<M::Field>,
        point: Option<M::Point>,
    ) -> M::Point {
        self.current_value = value;
        replace(&mut self.current_point, point).expect("Point not contained")
    }
}

pub struct FuncGrad<M, F>
where
    M: Manifold,
    F: Fn(&M::Point, bool, bool) -> (Option<M::Field>, Option<M::TangentVector>),
{
    manifold: M,
    fun_with_grad: F,
    current_point: Option<M::Point>,
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
            current_point: None,
            current_value: None,
            current_grad: None,
        }
    }

    pub fn new_with_fun_grad<Fun, Grad>(
        manifold: M,
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
            current_point: None,
            current_value: None,
            current_grad: None,
        }
    }
}

impl<M, F> FuncGrad<M, F>
where
    M: Manifold + EGradToRGrad,
    F: Fn(&M::Point, bool, bool) -> (Option<M::Field>, Option<M::TangentVector>),
{
    pub fn new_with_fun_egrad(
        manifold: M,
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
            current_point: None,
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

    fn update_value(&mut self, x: M::Point) -> M::Field {
        let value = (self.fun_with_grad)(&x, true, false).0;
        self.current_value = value;
        self.current_point = Some(x);
        self.current_grad = None;
        value.expect("Error")
    }

    fn get_value(&self) -> M::Field {
        self.current_value.expect("Update first")
    }

    fn get_point(&self) -> &M::Point {
        self.current_point.as_ref().expect("Point not contained")
    }

    fn replace_value_point(
        &mut self,
        value: Option<M::Field>,
        point: Option<M::Point>,
    ) -> M::Point {
        self.replace_value_point_gradient(value, point, None).0
    }

    fn compute_value(&self, x: &M::Point) -> M::Field {
        (self.fun_with_grad)(x, true, false).0.unwrap()
    }
}

impl<M, F> FuncOne for FuncGrad<M, F>
where
    M: Manifold,
    F: Fn(&M::Point, bool, bool) -> (Option<M::Field>, Option<M::TangentVector>),
{
    fn compute_value_gradient(&self, x: &M::Point) -> (M::Field, M::TangentVector) {
        let (value, grad) = (self.fun_with_grad)(&x, true, true);
        (value.expect("Error"), grad.expect("Error"))
    }

    fn update_value_gradient(&mut self, x: M::Point) -> M::Field {
        let (value, grad) = (self.fun_with_grad)(&x, true, true);
        self.current_value = value;
        self.current_grad = grad;
        self.current_point = Some(x);
        value.expect("Error")
    }

    fn get_gradient(&self) -> &M::TangentVector {
        self.current_grad.as_ref().expect("Not eval yet")
    }

    fn return_gradient(&mut self) -> M::TangentVector {
        replace(&mut self.current_grad, None).expect("Not eval yet")
    }

    fn replace_value_point_gradient(
        &mut self,
        value: Option<M::Field>,
        point: Option<M::Point>,
        grad: Option<M::TangentVector>,
    ) -> (M::Point, M::TangentVector) {
        self.current_value = value;
        let point = replace(&mut self.current_point, point).expect("Point not contained");
        let grad = replace(&mut self.current_grad, grad).expect("Gradient not contained");
        (point, grad)
    }

    fn directly_get_gradient(&self, x: &M::Point) -> M::TangentVector {
        (self.fun_with_grad)(x, false, true).1.unwrap()
    }
}

pub struct FuncGradHess<M, F>
where
    M: Manifold,
{
    manifold: M,
    fun_with_grad_hess: F,
    current_point: Option<M::Point>,
    current_tangent_vector: Option<M::TangentVector>,
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
            current_point: None,
            current_tangent_vector: None,
            current_value: None,
            current_grad: None,
            current_hess: None,
        }
    }

    pub fn new_with_fun_grad_hess<Fun, Grad, Hess>(
        manifold: M,
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
            current_point: None,
            current_tangent_vector: None,
            current_value: None,
            current_grad: None,
            current_hess: None,
        }
    }
}

impl<M> FuncGradHess<M, ()>
where
    M: Manifold + EGradToRGrad + EHessToRHess,
{
    pub fn new_from_ambient<F1>(
        manifold: M,
        fun_with_egrad_ehess: F1,
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
    >
    where
        F1: Fn(
            &M::Point,
            Option<&M::TangentVector>,
            bool,
            bool,
            bool,
        ) -> (
            Option<M::Field>,
            Option<M::AmbientPoint>,
            Option<M::AmbientPoint>,
        ),
    {
        let manifold_ = manifold.clone();
        let fun_with_grad_hess =
            move |x: &M::Point, v: Option<&M::TangentVector>, cv: bool, cg: bool, ch: bool| {
                let cg = cg || ch;
                let (cost, egrad, ehess) = fun_with_egrad_ehess(x, v, cv, cg, ch);

                let grad = if cg {
                    Some(manifold_.egrad_to_rgrad(x, egrad.as_ref().unwrap()))
                } else {
                    None
                };
                let hess = if ch {
                    Some(manifold_.ehess_to_rhess(x, v.unwrap(), &egrad.unwrap(), &ehess.unwrap()))
                } else {
                    None
                };
                (cost, grad, hess)
            };

        FuncGradHess {
            manifold,
            fun_with_grad_hess,
            current_point: None,
            current_tangent_vector: None,
            current_value: None,
            current_grad: None,
            current_hess: None,
        }
    }

    pub fn new_with_fun_egrad_ehess<Fun, Grad, Hess>(
        manifold: M,
        function: impl Fn(&M::Point) -> M::Field,
        egradient: impl Fn(&M::Point) -> M::AmbientPoint,
        ehessian: impl Fn(&M::Point, &M::TangentVector) -> M::AmbientPoint,
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
        let manifold_ = manifold.clone();
        let fun_with_grad_hess = move |x: &M::Point,
                                       v: Option<&M::TangentVector>,
                                       cv: bool,
                                       cg: bool,
                                       ch: bool| {
            let value = if cv { Some(function(x)) } else { None };

            let (grad, hess) = if cg || ch {
                let egrad = egradient(x);
                let grad = if cg {
                    Some(manifold_.egrad_to_rgrad(x, &egrad))
                } else {
                    None
                };
                let hess = if ch {
                    Some(manifold_.ehess_to_rhess(x, v.unwrap(), &egrad, &ehessian(x, v.unwrap())))
                } else {
                    None
                };

                (grad, hess)
            } else {
                (None, None)
            };
            (value, grad, hess)
        };

        FuncGradHess {
            manifold,
            fun_with_grad_hess,
            current_point: None,
            current_tangent_vector: None,
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

    fn compute_value(&self, x: &M::Point) -> M::Field {
        (self.fun_with_grad_hess)(x, None, true, false, false)
            .0
            .unwrap()
    }

    fn update_value(&mut self, x: M::Point) -> M::Field {
        let value = (self.fun_with_grad_hess)(&x, None, true, false, false).0;
        self.current_value = value;
        self.current_grad = None;
        self.current_hess = None;
        self.current_point = Some(x);
        self.current_tangent_vector = None;
        value.expect("Error")
    }

    fn get_value(&self) -> M::Field {
        self.current_value.expect("Update first")
    }

    fn get_point(&self) -> &M::Point {
        self.current_point.as_ref().expect("Point not contained")
    }

    fn replace_value_point(
        &mut self,
        value: Option<M::Field>,
        point: Option<M::Point>,
    ) -> M::Point {
        self.replace_value_point_gradient(value, point, None).0
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
    fn compute_value_gradient(&self, x: &M::Point) -> (M::Field, M::TangentVector) {
        let (value, grad, _) = (self.fun_with_grad_hess)(&x, None, true, true, false);
        (value.expect("Error"), grad.expect("Error"))
    }

    fn update_value_gradient(&mut self, x: M::Point) -> M::Field {
        let (value, grad, _) = (self.fun_with_grad_hess)(&x, None, true, true, false);
        self.current_value = value;
        self.current_grad = grad;
        self.current_hess = None;
        self.current_point = Some(x);
        self.current_tangent_vector = None;
        value.expect("Error")
    }

    fn get_gradient(&self) -> &M::TangentVector {
        self.current_grad.as_ref().expect("update first")
    }

    fn return_gradient(&mut self) -> M::TangentVector {
        replace(&mut self.current_grad, None).expect("update first")
    }

    fn replace_value_point_gradient(
        &mut self,
        value: Option<M::Field>,
        point: Option<M::Point>,
        grad: Option<M::TangentVector>,
    ) -> (M::Point, M::TangentVector) {
        self.current_value = value;
        self.current_hess = None;
        self.current_tangent_vector = None;
        let point = replace(&mut self.current_point, point).expect("Point not contained");
        let grad = replace(&mut self.current_grad, grad).expect("Gradient not contained");
        (point, grad)
    }

    fn directly_get_gradient(&self, x: &M::Point) -> M::TangentVector {
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
    fn compute_value_gradient_hessian(
        &mut self,
        x: &M::Point,
        v: &M::TangentVector,
    ) -> (M::Field, M::TangentVector, M::TangentVector) {
        let (value, grad, hess) = (self.fun_with_grad_hess)(x, Some(v), true, true, true);
        (
            value.expect("Error"),
            grad.expect("Error"),
            hess.expect("Error"),
        )
    }

    fn update_value_gradient_hessian(&mut self, x: M::Point, v: M::TangentVector) -> M::Field {
        let (value, grad, hess) = (self.fun_with_grad_hess)(&x, Some(&v), true, true, true);
        self.current_value = value;
        self.current_grad = grad;
        self.current_hess = hess;
        self.current_point = Some(x);
        self.current_tangent_vector = Some(v);
        value.expect("Error")
    }

    fn get_hessian(&self) -> &M::TangentVector {
        &self.current_hess.as_ref().expect("update first")
    }

    fn return_hessian(&mut self) -> M::TangentVector {
        replace(&mut self.current_hess, None).expect("update first")
    }

    fn directly_get_hessian(&self, x: &M::Point, v: &M::TangentVector) -> M::TangentVector {
        (self.fun_with_grad_hess)(x, Some(v), false, false, true)
            .2
            .unwrap()
    }
}
