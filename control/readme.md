# Control

## Pure Pursuit

**Local Frame**

This is the pure pursuit algorithm. It is a path tracking algorithm that is used to follow a path. The algorithm calculates the steering angle of the vehicle based on the current position of the vehicle and the path to be followed.

$$
R = \frac{L^2}{2 \cdot y_{ref}}
$$

Steering angle should be proportional to the curvature of the path.

$$
\alpha = \frac{1}{R} = \frac{2 \cdot |y_{ref}|}{L^2}
$$


**Inertia Frame**
Pure pursuit algorithm in the inertial frame is shown as below.

$$
R = \frac{L}{2 \cdot sin(\alpha)}
$$

Given a current speed $v_r$, the commanding heading rate is:

$$
\omega = \frac{v_r}{R} = \frac{2 v_r sin(\alpha)}{L}
$$

$$
tan(\alpha + \theta_r) = \frac{y_{ref} - y_r}{x_{ref} - x_r}

$$
where $\theta_r$ is the current heading of the car.

QUESTIONS:

- what is the default direction of steering? negative for left, positive for right?


**Procedure**

if using vehicle frame, 
- find a new pose ($x_r, y_r, \theta_r$) of the car
- find the path point cloest to the car: it is possible that there multiple points one lookahead distance away from the car, in this case, the closest one (the one with least distance along the path) is chosen.
- transform the goal point to the vehicle frame
- go towards that waypoint with calculated steering angle
- localize the car with the new pose

**Tuning**
the look ahead distance L is a parameter in the algorithm.
- A smaller L leads to more aggressive maneuvering to follow a closer arc, and closer arcs can work against the dynamic limits of the car
- A larger L leads to smoother maneuvering, but the car may not follow the path as closely, thus, leading to higher tracking errors.


The lookahead distance is usually chosen to be a function of the speed of the vehicle, so that $\omega$ will not become more sensitive to $\alpha$ when $v_r$ is higher. The higher the speed, the higher the lookahead distance. This is because at higher speeds, the vehicle will cover more distance in the time it takes to react to the path. Thus, the lookahead distance should be higher to ensure that the vehicle has enough time to react to the path.