"""Analytic valuation of a signed fixed-accrual cashflow on an IFR model."""

import numpy as np

from fixedincomelib.model import ModelType
from fixedincomelib.product import ProductFixedAccruedCashflow
from fixedincomelib.valuation import (
    AnalyticValParam, FundingIndexParameter, ValuationEngineProduct,
    ValuationEngineProductRegistry,
)


class ValuationEngineProductFixedAccrued(ValuationEngineProduct):
    def __init__(self, model, valuation_parameters_collection, product, request):
        super().__init__(model, valuation_parameters_collection, product, request)
        funding_parameter = valuation_parameters_collection.get_vp_from_build_method_collection(
            FundingIndexParameter._vp_type,
        )
        self.funding_index_ = funding_parameter.get_funding_index(product.currency)

    def calculate_value(self) -> None:
        product = self.product_
        self.value_ = self.cash_ = 0.0
        if product.payment_date < self.value_date:
            return

        amount = product.notional * product.accrued
        if product.payment_date == self.value_date:
            self.value_ = self.cash_ = amount
        else:
            self.value_ = amount * self.model.discount_factor(
                self.funding_index_, product.payment_date
            )

    def calculate_risk(
        self, gradient: list[np.ndarray], scaler: float = 1.0, accumulate: bool = False,
    ) -> None:
        product = self.product_
        if product.payment_date <= self.value_date:
            self.model.resize_gradient(gradient)
            if not accumulate:
                for block in gradient:
                    block.fill(0.0)
            return

        self.model.discount_factor_gradient_wrt_state(
            self.funding_index_,
            product.payment_date,
            gradient,
            scaler=scaler * product.notional * product.accrued,
            accumulate=accumulate,
        )


ValuationEngineProductRegistry().register(
    (ModelType.YIELD_CURVE.to_string(), ProductFixedAccruedCashflow._product_type, AnalyticValParam._vp_type),
    ValuationEngineProductFixedAccrued,
)
