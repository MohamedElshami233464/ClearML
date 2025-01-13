import time

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0, sample_time=0.01, output_limits=(None, None)):
        """
        Initializes the PID controller.
        
        :param kp: Proportional gain
        :param ki: Integral gain
        :param kd: Derivative gain
        :param setpoint: Target value
        :param sample_time: Time between updates (seconds)
        :param output_limits: Min and max output values (tuple)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.sample_time = sample_time
        self.output_limits = output_limits

        self._prev_error = 0
        self._integral = 0
        self._last_time = None

    def reset(self):
        """Resets the PID controller."""
        self._prev_error = 0
        self._integral = 0
        self._last_time = None

    def compute(self, feedback_value):
        """
        Computes the PID output.

        :param feedback_value: Current value to compare against the setpoint
        :return: PID output
        """
        current_time = time.time()
        error = self.setpoint - feedback_value
        delta_time = (current_time - self._last_time) if self._last_time else self.sample_time

        if self._last_time is None:
            self._last_time = current_time
            return 0

        # Proportional term
        proportional = self.kp * error

        # Integral term
        self._integral += error * delta_time
        integral = self.ki * self._integral

        # Derivative term
        derivative = self.kd * (error - self._prev_error) / delta_time

        # Compute output
        output = proportional + integral + derivative

        # Clamp output to limits
        if self.output_limits[0] is not None:
            output = max(self.output_limits[0], output)
        if self.output_limits[1] is not None:
            output = min(self.output_limits[1], output)

        # Save error and time for the next update
        self._prev_error = error
        self._last_time = current_time

        return output
