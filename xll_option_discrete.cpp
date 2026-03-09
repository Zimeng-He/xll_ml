// xll_option_discrete.cpp
#include "fms_option_discrete.h"
#include "xll_ml.h"

#undef CATEGORY
#define CATEGORY L"OPTION"

using namespace xll;
using namespace fms::option;

AddIn xai_option_discrete_(
    Function(XLL_HANDLEX, L"xll_option_discrete_", L"\\" CATEGORY L".DISCRETE")
    .Arguments({
        Arg(XLL_FP, L"x", L"is the array of support values."),
        Arg(XLL_FP, L"p", L"is the array of probabilities.")
        })
    .Uncalced()
    .Category(CATEGORY)
    .FunctionHelp(L"Return handle to discrete option pricing model.")
);

HANDLEX WINAPI xll_option_discrete_(_FP12* x, _FP12* p)
{
#pragma XLLEXPORT
    HANDLEX result = INVALID_HANDLEX;

    try {
        ensure(size(*x) == size(*p) || !"x and p must have same size");
        handle<base<>> h_(new discrete::model<>(size(*x), x->array, p->array));
        ensure(h_);

        result = h_.get();
    }
    catch (const std::exception& ex) {
        XLL_ERROR(ex.what());
    }
    catch (...) {
        XLL_ERROR(__FUNCTION__ ": unknown exception");
    }

    return result;
}

AddIn xai_option_discrete(
	Function(XLL_FP, L"xll_option_discrete", CATEGORY L".DISCRETE")
	.Arguments({ Arg(XLL_HANDLEX, L"h", L"is the handle returned by \\OPTION.DISCRETE.")
		})
	.Category(CATEGORY)
	.FunctionHelp(L"Return normalized xi values")
);
_FP12* WINAPI xll_option_discrete(HANDLEX h)
{
#pragma XLLEXPORT
    static FPX out;
    try {
        handle<base<>> h_(h);
        ensure(h_);
        auto* d = h_.as<discrete::model<>>();
        ensure(d);
        const auto& x = d->values();
        
        out.resize((int)x.size(), 1);
        for (int i = 0; i < (int)x.size(); ++i) {
            out[i] = x[i];
        }
    }

    catch (const std::exception& ex) {
        XLL_ERROR(ex.what());
    }
    catch (...) {
        XLL_ERROR(__FUNCTION__ ": unknown exception");
        return nullptr;
    }

    return out.get();
}