// xll_ml.cpp
#include "fms_perceptron.h"
#include "xll24/include/xll.h"

using namespace xll;

// Create mdspan from FP data type.
constexpr auto make_mdspan(_FP12* pa)
{
	return std::experimental::mdspan<double, std::extents<INT32, std::dynamic_extent>>
		(pa->array, pa->rows * pa->columns);
}
constexpr auto make_mdspan(const _FP12* const pa)
{
	return std::experimental::mdspan<const double, std::extents<INT32, std::dynamic_extent>>
		(pa->array, pa->rows * pa->columns);
}