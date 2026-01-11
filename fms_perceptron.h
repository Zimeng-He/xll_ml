// fms_perceptron.h
// A perceptron is a hyperplane separating two sets of points in R^n.
// Given sets S_0 and S_1, find a vector w and a scalar b such that
// w.x + b < 0 for x in S_0 and w.x + b > 0 for x in S_1.
#pragma once

#include <exception>
#include <span>
#define MDSPAN_USE_PAREN_OPERATOR 1
#include <experimental/mdspan>
#include <experimental/linalg>

namespace fms::perceptron {

    // Update weights w given point x and label y in {-1, 1}
    template<class T>
    T update(std::span<T>& _w, const std::span<T>& _x, int y, T alpha = 1.0)
    {
        using std::experimental::linalg::add;
        using std::experimental::linalg::dot;
        using std::experimental::linalg::scaled;

        if (_w.size() != _x.size()) {
            throw std::invalid_argument("weight and point must have the same size");
        }
        if (y != 1 and y != -1) {
            throw std::invalid_argument("label must be 1 or -1");
        }

		std::mdspan w(_w.data(), _w.size());
		std::mdspan x(_x.data(), _x.size());

		double y_ = dot(w, x);
        // Check if misclassified
        if (y_ * y < 0) {
            // Update: w = w + alpha dy x    ,
            add(w, scaled(alpha * (y_ - y), x), w);
        }
        
        return y_;
    }

    template<class T = double>
    struct neuron {
        std::vector<T> w;

        neuron(size_t n)
            : w(n)
		{ }
        neuron(const std::vector<T>& w)
            : w(w)
        { }
        neuron(const neuron&) = default;
        neuron& operator=(const neuron&) = default;
        neuron(neuron&&) = default;
        neuron& operator=(neuron&&) = default;
        ~neuron() = default;

        void update(const std::span<const T>& x, int y, double alpha = 1.0)
        {
            if (x.size() == 0 or x[0] != 1) {
                throw std::invalid_argument("first element of data must be 1");
            }
  
            fms::perceptron::update(w, x, y, alpha);
		}
        
        using pair = std::pair<const std::span<const T>, int>;
        void train(const std::span<pair>& xy, double alpha = 1.0)
        {
            for (const auto [x, y] : xy) {
                update(x, y, alpha);
            }
        }
    };
 
} // namespace fms::perceptron
