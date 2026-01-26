// fms_perceptron.h
// A perceptron is a hyperplane separating two sets of points in R^n.
// Given sets S_0 and S_1, find a vector w and a scalar b such that
// w.x < 0 for x in S_0 and w.x > 0 for x in S_1.
#pragma once

#include <vector>
#include "fms_error.h"
#include "fms_linalg.h"

namespace fms::perceptron {

    // Update weights w given point x and label y in {true, false}
    template<class T = double>
    bool update(std::span<T>& _w, const std::span<T>& _x, bool y, T alpha = 1.0)
    {
        ensure (_w.size() == _x.size() || !"weight and point must have the same size");
        //ensure (y == 0 or y == 1 || !"label must be 0 or 1");

		std::span<T> w(_w.data(), _w.size());
		std::span<T> x(_x.data(), _x.size());

		bool y_ = fms::linalg::dot(w, x) > 0;
        // Check if misclassified
        if (y_ * y < 0) {
            // Update: w = w + alpha dy x    ,
            fms::linalg::axpy(alpha * (y - y), x, w, w);
        }
        return y == y_;
    }
    /*
    template<class T = double>
    bool train(const std::span<const T>& x, bool y, T alpha = 1.0, std::size_t n = 100)
    {
        while (n && false == fms::perceptron::update(w, x, y, alpha)) {
            --n; // limit loops
        }

        return n != 0;
    }
    */

    template<class T = double>
    struct neuron {
        std::vector<T> w;

        neuron(size_t n)
            : w(n)
		{ }
        // RAII
        neuron(std::span<T> w)
            : w(w)
        { }
        neuron(const neuron&) = default;
        neuron& operator=(const neuron&) = default;
        neuron(neuron&&) = default;
        neuron& operator=(neuron&&) = default;
        ~neuron() = default;

        void update(const std::span<const T>& x, int y, double alpha = 1.0, std::size_t n = 100)
        {
            fms::perceptron::update(w, x, y, alpha);
		}
        bool train(const std::span<const T>& x, bool y, T alpha = 1.0, std::size_t n = 100)
        {
            while (n && false == fms::perceptron::update(w, x, y, alpha)) {
                --n; // limit loops
            }
            
            return n != 0;
        }

        using pair = std::pair<const std::span<const T>, int>;
        void train(const std::span<pair>& xy, double alpha = 1.0)
        {
            for (const auto [x, y] : xy) {
                train(x, y, alpha);
            }
        }
    };
 
} // namespace fms::perceptron