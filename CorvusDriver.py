# This file is part of PyLabSAS.
# Original author: Raphael Viera (raphael.viera@emse.fr).
# Adapted by Hugo Perrin to remove PylabSAS dependency (h.perrin@emse.fr) (16/05/2023)
#
# MIT License
#
# Copyright (c) 2021 Raphael Viera, Ecole des Mines de Saint-Etienne - Campus G. Charpak - Gardanne, France
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
This module implements the methods to communicate with the instrument through serial com.
"""

# Built-in modules
from enum import Enum
import time

# Third-party modules
import serial
from serial import SerialException


class Axes(Enum):
    """ Possible axes. """
    VIRTUAL = 0
    X = 1
    Y = 2
    Z = 3

class Unit(Enum):
    """ Possible units. """
    MICROSTEP = 0 # 1 motor revolution / 40000 steps.
    MICROMETER = 1
    MILIMETER = 2
    CENTIMETER = 3
    METER = 4
    INCH = 5
    MIL = 6 # 1/100 inch

class Status(Enum):
    """ Possible status. """
    READY = 0
    CURRENT_COMMAND_EXEC = 1
    JOYSTICK_ACTIVE = 2
    BUTTON_A = 4
    MACHINE_ERROR = 8
    SPEED_MODE = 16
    IN_WINDOW = 32
    SETINFUNC = 64
    MOTOR_ENABLE_SAFETY_DEFICE = 128
    JOYSTICK_BUTTON = 256


class SMCCorvusXYZ():
    """
    Please check the correspondent .md file for further details.
    """

    def __init__(self, port=None, baud_rate=None, bytesize=None, stopbits=None, parity=None, timeout=None, accel: int = 1000, velocity: int = 800):

        self.ser = None

        self.port = port
        self.baud_rate = baud_rate
        self.bytesize = bytesize
        self.stopbits = stopbits
        self.parity = parity
        self.timeout = timeout

        self.acceleration = accel # um/s^2
        self.velocity = velocity # um/s
        self.unit = Unit.MICROMETER

        self.axis_range = [-50000, 50000]  # unit. e.g. 50000 micrometers

        self.open_instrument()
        self.set_defaults()

    def open_instrument(self):
        """
        Open the instrument from the specified self.PORT.

        Args:
            - port (string): the communication port.
            - baud_rate (int): bits per seconds (baud rate).

        Returns:
            - None.
        """

        kwargs= {'port':     self.port,
                 'baudrate': self.baud_rate,
                 'bytesize': getattr(serial, self.bytesize),
                 'stopbits': getattr(serial, self.stopbits),
                 'parity':   getattr(serial, self.parity),
                 'timeout':  self.timeout}
        try:
            self.ser = serial.Serial(**kwargs)
        except:
            print(f"Error: Could not open serial port: {self.port}")

        print(f"Debug: Serial port is open: {self.port}")

    def close_instrument(self):
        try:
            self.ser.close()
        except:
            pass

    def __del__(self):
        self.close_instrument()

    def set_defaults(self):

        print("Debug: Setting XYZ table defaults.")
        # Define value=None so the program reads the default value.
        for axis in ['VIRTUAL', 'X', 'Y', 'Z']:
            self.set_unit(None, axis)
        self.set_velocity() # 200 um/s
        self.set_acceleration() # 200 um/s
        self.save_configs()

    def write(self, cmd):
        """
        Send a command to the instrument.

        Args:
            - cmd (string): the command to be sent to the instrument.

        Returns:
            - None.
        """

        data = self.ser.write(str.encode(f"{cmd}\n"))

    def read(self):
        """
        Read from the instrument the last written value.

        Args:
            -None.

        Returns:
            - data_s (string): the value read.
        """

        data_b = self.ser.read(100)
        data_s = data_b.decode()
        return data_s

    def reset_instrument(self):
        """
        The command reset preforms a device reset which is equal to disconnect the device from the power.
        The proper state of the controller after a reset is indicated with beep (1s).
        """

        self.write('reset')
        # result = self.read()
        print("Debug: Resetting instrument.")
        time.sleep(15)

    def get_error(self):
        """
        With the command geterror the last occurred system error is returned.
        Afterwards the error code memory is cleared.
        The occurrence of an system error is not reflected in the status reply.

        Returns:
            - result (string): the error code (see manual).
        """

        self.write('ge')
        result = self.read()
        print(f"Debug: Get_error returned: {result}")
        return result

    def save_configs(self):
        """
        The command save stores all active parameters in a non volatile memory.
        Always the last saved settings are restored after power on.
        """

        self.write('save')
        result = self.read()
        print(f"Debug: Save configs into NVM: {result}")

    def restore_configs(self):
        """
        The command restore reactivates the last saved parameters.  With the command sequence restore save
        the controller replies a status information after the restore is finished.
        """

        self.write('restore')
        result = self.read()
        print(f"Debug: Restore configs from NVM: {result}")

    def get_status(self):
        """
        Command status returns the current state of the controller.
        Each state is assigned to a binary digit from D0 to D8.
        If more states are active, the decimal values of the digits are added.
        To decode the replied status, it is necessary to convert the decimal value into a binary pattern and
        mask the bits.

        Returns:
            - result (int): the status code (see manual).
        """

        self.write('st')
        result = self.read()

        try:
            result = int(result)
            print(f"Debug: Status: {Status(result)}")
        except ValueError:
            err = self.get_error()
            print(f"Error: get_status(): Unexpected response from the instrument: {err}.")
        return result

    def set_joystick(self, set_on: bool):
        """
        The command joystick enables or disables the manual mode.
        """

        self.write(f'{int(set_on)} j')

    def get_acceleration(self):
        """
        The command getaccel (ga) returns the setting of setaccel.

        Returns:
            - result (string): the acceleration value.
        """

        self.write("ga")
        result = self.read()
        print(f"Debug: Get_acceleration returned: {result}")
        return result

    def set_acceleration(self, val=None):
        """
        Command set accel (sa) defines the acceleration ramp with which the controller executes the programmed move.
        The axes are linear interpolated, this means the controller starts and stops all axes simultaneously.
        The value of set accel relates to the axis which must travel the longest distance.
        The maximum acceleration of the other axes depends on the ratio to the axis with the longest travel.
        Acceleration and deceleration ramp are identical.

        Args:
            - val (int): the acceleration value
        """
        if val is None:
            val = self.acceleration

        self.write(f"{val} sa")
        print(f"Debug: Set acceleration: {val}")

    def get_velocity(self):
        """
        The command getvel (gv) returns the setting of set vel.

        Returns:
            - result (string): the velocity value.
        """

        self.write("gv")
        result = self.read()
        print(f"Debug: Get_velocity returned: {result}")
        return result
############################################################################## STOPPED HERE
    def set_velocity(self, val=None):
        """
        Command setvel configures the programmed move velocity va.
        In consideration to the given move distances of all active axes, the controller calculates an individual
        velocity profile for each axis.
        The setting of setvel relate to the axis, that moves the longest distance, see diagram.
        The maximum velocity Vb or Vc depends on the distance ratio to the axis with the longest travel.
        minimum velocity: 15,26 nm/s
        maximum velocity: 45 rev./s, pitch =4 mm -> 180mm/s
                          60 rev./s (option)

        Args:
            - val (float): velocity of all axes.
        """
        if val is None:
            val = self.velocity

        self.write(f'{val} sv')
        print(f"Debug: Set velocity: {val}")

    def get_unit(self, axis: Axes):
        """
        The command getunit returns the settings the physical units.

        Args:
            - axis (Axes): the axis in question.

        Returns:
            - result (string): the physical unit.
        """
        self.write(f'{axis} getunit')
        result = self.read()

        print(f"Debug: Get_unit returned: {result}")
        return result

    def set_unit(self, val: Unit = None, axis: Axes = None):
        """
        With command setunit the physical units of the Axis specific parameters are defined.
        The units of velocity and acceleration are determined from the unit setting of Axis-0.
        The unit of the commands setcalvel, setncalvel, setrmvel, setnrmvel and setrefvel are fixed to rev./s (r/s).
        For the reason of compatibility with older controllers, the unit microstep is emulated from Corvus.
        In this case the positioning resolution is reduced.

        By changing the unit on the "VIRTUAL" axis, the following commands are influenced: setvel, setaccel,
        setmanaccel.
        By changing the unit on the "X", "Y" and "Z" axes, the following commands are influenced: move, rmove, pos,
        setpos, setpitch, setlimit, setcalswdist, setclperiod.

        Args:
            - val (Unit): The physical unit.
            - axis (Axes): The physical axis.
        """
        if val is None:
            val = self.unit
        if axis is None:
            axis = Axes.VIRTUAL

        self.write(f'{val} {axis} setunit')
        print(f"Debug: Set unit: {val} to axis {axis}")

    def get_pos(self, return_list=False):
        """
        Command pos return the current coordinate of all active axes.
        The position value relates to the origin which is defined with command cal or setpos.
        The number of replied coordinates depends on the setting of setdim.

        Args:
            - return_list (boolean): option to return list or string.

        Returns:
            - result (string): the coordinate of active axes.
        """

        self.write(f'pos')
        result = self.read()
        try:
            result_list = [float(pos) for pos in result.split()]  # e.g. [1.0, 1.0, 1.0]
        except:
            print("Error: get_pos(): Unexpected response from the instrument.")

        print(f"Debug: Get_pos returned: X Y Z > {result}")

        if return_list:
            result = result_list
        return result

    def set_pos(self, x_axis=None, y_axis=None, z_axis=None):
        """
        With command setpos the point of origin of all axes can be defined.
        The coordinates of the limits will be recalculated if the point origin changes.
        The axes must be enabled.
        For special cases the zero point can be defined with a relative offset.

        Examples:
        0 0 0 setpos
        The current coordinate is defined as the point of origin.

        10 10 10 setpos / unit = mm
        The current coordinate is defined as the point of origin with a relative offset 10 mm each axis.
        The command pos will reply the position value -10 -10 -10 if the previous coordinate was 0 0 0.

        Args:
            - x_axis (float): the x-axis point of origin.
            - y_axis (float): the y-axis point of origin.
            - z_axis (float): the z-axis point of origin.

        Returns:
            - result (string): the x y z coordinates.
        """
        x_axis = max(self.axis_range[0], min(self.axis_range[1], x_axis))
        y_axis = max(self.axis_range[0], min(self.axis_range[1], y_axis))
        z_axis = max(self.axis_range[0], min(self.axis_range[1], z_axis))

        self.write(f'{x_axis} {y_axis} {z_axis} setpos')
        result = self.read()
        print(f"Debug: Set pos: {x_axis}, {y_axis}, {z_axis}")
        return result

    def set_zero(self):
        """
        Set the origin of all axes to 0 (zero).
        """
        print("Info: Setting set_zero()")
        self.set_pos(0, 0, 0)
        self.save_configs()


    def move_x_abs(self, x_axis: float):
        """
        Move X-axis absolute.

        Args:
            - x_axis (float): distance to travel in the X axis.
        """
        x_axis = max(self.axis_range[0], min(self.axis_range[1], x_axis))

        curr_pos_y = self.get_pos_y()
        curr_pos_z = self.get_pos_z()
        self.move_xyz_abs(x_axis, curr_pos_y, curr_pos_z)

    def move_y_abs(self, y_axis: float):
        """
        Move Y-axis absolute.

        Args:
            - y_axis (float): distance to travel in the Y axis

        """
        y_axis = max(self.axis_range[0], min(self.axis_range[1], y_axis))

        curr_pos_x = self.get_pos_x()
        curr_pos_z = self.get_pos_z()
        self.move_xyz_abs(curr_pos_x, y_axis, curr_pos_z)

    def move_z_abs(self, z_axis: float):
        """
        Move Z-axis absolute.

        Args:
            - z_axis (float): distance to travel in the Z axis.

        """
        z_axis = max(self.axis_range[0], min(self.axis_range[1], z_axis))

        curr_pos_x = self.get_pos_x()
        curr_pos_y = self.get_pos_y()
        self.move_xyz_abs(curr_pos_x, curr_pos_y, z_axis)

    def move_xyz_abs(self, x_axis: float, y_axis: float, z_axis: float):
        """
        Command move executes point to point positioning tasks to absolute coordinates based on the point of origin.
        The move profile is calculated in respect to the velocity/acceleration setup and the given hard or
        software limits. The axes are linear interpolated, this causes the controller to start and stop all active
        axes simultaneously.
        Command status returns the actual state of the move procedure.

        Args:
            - x_axis (float): distance to travel in the X axis.
            - y_axis (float): distance to travel in the Y axis.
            - z_axis (float): distance to travel in the Z axis.
        """
        x_axis = max(self.axis_range[0], min(self.axis_range[1], x_axis))
        y_axis = max(self.axis_range[0], min(self.axis_range[1], y_axis))
        z_axis = max(self.axis_range[0], min(self.axis_range[1], z_axis))

        x_axis = -x_axis # Negative so it can go from left to right by default.

        self.write(f'{x_axis} {y_axis} {z_axis} m')

        # Wait until movement finishes
        print("Info: Moving...")

        self.wait_move_finish(x_axis, y_axis, z_axis)

        print("Debug: Ready")

    def wait_move_finish(self, x_axis: float, y_axis: float, z_axis: float):
        sleep_time = self.timeout/2
        sum_sleep_time = 0
        timeout = 30
        temp_status = None

        while temp_status not in (Status.READY.value, Status.JOYSTICK_ACTIVE.value):
            time.sleep(sleep_time)
            temp_status = self.get_status()
            print(f"Debug: Status: {temp_status}")
            sum_sleep_time += sleep_time
            if sum_sleep_time >= timeout:
                err = self.get_error()
                self.reset_instrument()
                self.restore_configs()
                self.set_defaults()
                self.reset_serial_comm()
                self.move_xyz_abs(x_axis, -y_axis, -z_axis)
                self.set_pos(0, 0, 0)
                print(f"Error: wait_move_finish(): Instrument not responding correctly. Error: {err}")

    def reset_serial_comm(self):
        """
        Reset serial communication with the instrument.
        """

        print("Warning: Resetting serial comm.")
        self.close_instrument()
        time.sleep(0.5)
        self.open_instrument()

    def move_x_rel(self, x_axis: float):
        """
        Move the X-axis relative.

        Args:
            - x_axis (float): distance to travel in the X axis.
        """
        x_axis = max(self.axis_range[0], min(self.axis_range[1], x_axis))

        self.move_xyz_rel(x_axis, 0, 0)

    def move_y_rel(self, y_axis: float):
        """
        Move the Y-axis relative.

        Args:
            - y_axis (float): distance to travel in the Y axis.
        """
        y_axis = max(self.axis_range[0], min(self.axis_range[1], y_axis))

        self.move_xyz_rel(0, y_axis, 0)

    def move_z_rel(self, z_axis: float):
        """
        Move the Z-axis relative.

        Args:
            - z_axis (float): distance to travel in the Z axis.
        """
        z_axis = max(self.axis_range[0], min(self.axis_range[1], z_axis))

        self.move_xyz_rel(0, 0, z_axis)

    def move_xyz_rel(self, x_axis: float, y_axis: float, z_axis: float):
        """
        Command rmove executes point to point positioning tasks relative to the current position.
        The move profile is calculated in respect to the velocity/acceleration setup and the given hard or
        software limits. The axes are linear interpolated, this causes the controller to start and stop all active
        axes simultaneously,
        The command status returns the actual state of the move procedure.

        Args:
            - x_axis (float): distance to travel in the X axis.
            - y_axis (float): distance to travel in the Y axis.
            - z_axis (float): distance to travel in the Z axis.
        """
        x_axis = max(self.axis_range[0], min(self.axis_range[1], x_axis))
        y_axis = max(self.axis_range[0], min(self.axis_range[1], y_axis))
        z_axis = max(self.axis_range[0], min(self.axis_range[1], z_axis))

        x_axis = -x_axis # Negative so it can go from left to right by default.

        self.write(f'{x_axis} {y_axis} {z_axis} r')

        # Wait until movement finishes
        print("Info: Moving...")

        self.wait_move_finish(x_axis, y_axis, z_axis)

        print("Debug: Ready")

    def getpitch(self, axis: Axes):
        self.write(f"{axis} getpitch")
        result = self.read()

        print(f"Debug: Get pitch on axis {axis}: {result}")
        return result

    def setpitch(self, value: float, axis: Axes):
        self.write(f"{value} {axis} setpitch")

