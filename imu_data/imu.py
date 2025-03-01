from pymavlink import mavutil

# Connect to the BlueROV via UDP
master = mavutil.mavlink_connection('udp:192.168.2.1:14550')

# Wait for a heartbeat to confirm connection
master.wait_heartbeat()
print("Connected to BlueROV. Streaming IMU data...\n")

# Open a file to save IMU data
with open("imu_data.txt", "w") as file:
    file.write("Time (us), Accel_X (mG), Accel_Y (mG), Accel_Z (mG), "
               "Gyro_X (mRad/s), Gyro_Y (mRad/s), Gyro_Z (mRad/s), "
               "Mag_X (mGauss), Mag_Y (mGauss), Mag_Z (mGauss)\n")

    while True:
        msg = master.recv_match(type='RAW_IMU', blocking=True)
        if msg:
            log_entry = (f"{msg.time_usec}, {msg.xacc}, {msg.yacc}, {msg.zacc}, "
                         f"{msg.xgyro}, {msg.ygyro}, {msg.zgyro}, "
                         f"{msg.xmag}, {msg.ymag}, {msg.zmag}\n")

            print(log_entry.strip())  # Print to console
            file.write(log_entry)  # Write to file
            file.flush()  # Ensure data is written immediately
