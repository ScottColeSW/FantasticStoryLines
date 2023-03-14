import math
import turtle
from piestimations import compute_pi

# # Define the compute_pi() function to calculate Pi to N digits
# def compute_pi(N):
#     pi_str = str(math.pi)
#     return pi_str[:N+2]

# Define the fractal function that draws the fractal
def fractal(length, depth, scale_factor, angle, pi_digit):
    # Base case
    if depth == 0:
        return

    # Define the turtle's movement
    turtle.forward(length)
    turtle.left(angle * int(float(pi_digit)))
    fractal(length * scale_factor, depth - 1, scale_factor, angle, str(compute_pi(depth))[:int(float(pi_digit)+2)])

    turtle.right(angle * 2 * int(float(pi_digit)))
    fractal(length * scale_factor, depth - 1, scale_factor, angle, str(compute_pi(depth))[:int(float(pi_digit)+2)])

    turtle.left(angle * float(pi_digit))
    turtle.backward(length)

# Set up the turtle's starting position and pen
turtle.speed(0)
turtle.penup()
turtle.goto(-200, 200)
turtle.pendown()

# Draw the fractal
# fractal(length, depth, scale_factor, angle, pi_digit)
fractal(150, 28, 0.6, 33, 100)

# Hide the turtle
turtle.hideturtle()

# Keep the window open until it is closed manually
turtle.mainloop()


