def getIndex(x, xs):
    for i in range(1, len(xs)):
        if x <= xs[i]:
            return i - 1

def firstDerivative(x, xs, ys):
    i = getIndex(x, xs)
    return (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) + ((ys[i + 2] - ys[i + 1]) / (xs[i + 2] - xs[i + 1]) - (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])) / (xs[i + 2] - xs[i]) * (2 * x - xs[i] - xs[i + 1])

def secondDerivative(x, xs, ys):
    i = getIndex(x, xs)
    return 2 * ((ys[i + 2] - ys[i + 1]) / (xs[i + 2] - xs[i + 1]) - (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])) / (xs[i + 2] - xs[i])

x = 0.2
xs = [-0.2, 0.0, 0.2, 0.4, 0.6]
ys = [-0.40136, 0.0, 0.40136, 0.81152, 1.2435]
# xs = [0.0, 0.1, 0.2, 0.3, 0.4]
# ys = [1.0,1.1052,1.2214,1.3499,1.4918]

print(f"Первая производная: {firstDerivative(x, xs, ys)}")
print(f"Вторая производная: {secondDerivative(x, xs, ys)}")