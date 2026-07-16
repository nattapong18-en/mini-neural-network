#include <cmath>
#include <iostream>

using namespace std;

double sigMoid(double z) { return 1.0 / (1.0 + exp(-z)); }

int main() {
  double x1{1.0};
  double x2{0.20};

  double w1{0.60};
  double w2{0.90};
  double w3{0.00};
  double w4{0.70};
  double w5{1.00};
  double w6{0.50};

  // h1
  double z = x1 * w1 + x2 * w4;

  // sigMoid
  sigMoid(z);
  cout << "h1_net: " << z << '\n';
  cout << "h1_out: " << sigMoid(z) << '\n';
  return 0;
}
