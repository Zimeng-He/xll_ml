// xll_ml.cpp
#include "fms_perceptron.h"
#undef ensure
#include "xll24/include/xll.h"

using namespace xll;

AddIn xai_perceptron_update(
	Function(XLL_FP, L"xll_perceptron_update", L"PERCEPTRON.UPDATE")
	.Arguments({
		Arg(XLL_FP, L"w", L"is an array of weights"),
		Arg(XLL_FP, L"x", L"is an array representing the input vector"),
		Arg(XLL_BOOL, L"y", L"is the label"),
		Arg(XLL_DOUBLE, L"alpha", L"is the learning rate (default=1.0)", 1.0)
		})
	.FunctionHelp(L"Update perceptron weights input vector and label.")
	.Category(L"ML")
);
_FP12* WINAPI xll_perceptron_update(_FP12* pw, _FP12*  px, BOOL y, double alpha)
{
#pragma XLLEXPORT
	try {
		alpha = alpha ? alpha : 1;
		auto w = span(*pw);
		auto x = span(*px);

		fms::perceptron::update(w, x, y, alpha);
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}
	catch (...) {
		XLL_ERROR("xll_perceptron_update: unknown exception");
	}
	return pw;
}

/*
AddIn xai_perceptron_train(
	Function(XLL_FP, L"xll_perceptron_train", L"PERCEPTRON.TRAIN")
	.Arguments({
		Arg(XLL_FP, L"w", L"is an array of weights"),
		Arg(XLL_FP, L"x", L"is an array representing the input vector"),
		Arg(XLL_BOOL, L"y", L"is the label"),
		Arg(XLL_DOUBLE, L"alpha", L"is the learning rate (default=1.0)", 1.0)
		})
	.FunctionHelp(L"Treain perceptron weights input vector and label.")
	.Category(L"ML")
);
_FP12* WINAPI xll_perceptron_train(_FP12* pw, _FP12* px, BOOL y, double alpha)
{
#pragma XLLEXPORT
	try {
		alpha = alpha ? alpha : 1;
		auto w = span(*pw);
		auto x = span(*px);

		fms::perceptron::train(w, x, y, alpha);
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}
	catch (...) {
		XLL_ERROR("xll_perceptron_update: unknown exception");
	}
	return pw;
}
*/