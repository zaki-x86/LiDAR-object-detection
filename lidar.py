import serial
import math
import numpy as np


class LidarData:
    def __init__(self, FSA, LSA, CS, Speed, TimeStamp, Confidence_i, Angle_i, Distance_i, Angle_norm, Distance_norm):
        self.FSA = FSA
        self.LSA = LSA
        self.CS = CS
        self.Speed = Speed
        self.TimeStamp = TimeStamp

        self.Confidence_i = Confidence_i
        self.Angle_i = Angle_i
        self.Distance_i = Distance_i
        self.Angle_norm = Angle_norm
        self.Distance_norm = Distance_norm


def CalcLidarData(str):
    str = str.replace(' ', '')

    minus_311 = (360 - 31.1) * math.pi / 180
    plus_311 = 31.1 * math.pi / 180

    Speed = int(str[2:4] + str[0:2], 16) / 100
    FSA = float(int(str[6:8] + str[4:6], 16)) / 100
    LSA = float(int(str[-8:-6] + str[-10:-8], 16)) / 100
    TimeStamp = int(str[-4:-2] + str[-6:-4], 16)
    CS = int(str[-2:], 16)

    Confidence_i = list()
    Angle_i = list()
    Distance_i = list()
    Angle_norm = list()
    Distance_norm = list()

    count = 0
    if (LSA - FSA > 0):
        angleStep = float(LSA - FSA) / (12)

    else:
        angleStep = float((LSA + 360) - FSA) / (12)

    counter = 0
    circle = lambda deg: deg - 360 if deg >= 360 else deg
    for i in range(0, 6 * 12, 6):
        Distance_i.append(int(str[8 + i + 2:8 + i + 4] + str[8 + i:8 + i + 2], 16) / 10)
        Confidence_i.append(int(str[8 + i + 4:8 + i + 6], 16))
        Angle_i.append(circle(angleStep * counter + FSA) * math.pi / 180.0)  # deg to rad
        counter += 1

    # for i in Angle_i:
    #     if i <= plus_311 or i >= minus_311:
    #         if Distance_i[Angle_i.index(i)] >= 50:
    #             Distance_norm.append(Distance_i[Angle_i.index(i)])
    #             Angle_norm.append(i)

    for i in Angle_i:
        if i <= plus_311:
            Distance_norm.append(Distance_i[Angle_i.index(i)])
            Angle_norm.append(0.5 + (i / plus_311) * 0.5)

        elif i >= minus_311:
            Distance_norm.append(Distance_i[Angle_i.index(i)])
            Angle_norm.append(((i - minus_311) / plus_311) * 0.5)
            print(i, (i / (2*math.pi)) * 0.5, Distance_i[Angle_i.index(i)])

    # Angle_norm = (np.array(Angle_norm) + plus_311).tolist()
    # if len(Angle_norm) >= 2:
    #     # Angle_norm = (1 - (np.array(Angle_norm) - np.min(Angle_norm)) / (np.max(Angle_norm) - np.min(Angle_norm))).tolist()
    #     # Angle_norm = ((np.array(Angle_norm) - np.min(Angle_norm)) / (np.max(Angle_norm) - np.min(Angle_norm))).tolist()
    #     Angle_norm = (1 - (np.array(Angle_norm) - plus_311) / (minus_311 - plus_311)).tolist()

    lidarData = LidarData(FSA, LSA, CS, Speed, TimeStamp, Confidence_i, Angle_i, Distance_i, Angle_norm, Distance_norm)
    return lidarData


class LidarNormData:
    def __init__(self, angles_norm, distances_norm, angles, distances):
        self.angles_norm = angles_norm
        self.distances_norm = distances_norm
        self.angles = angles
        self.distances = distances


def lidarfunc(lidar_time):
    ser = serial.Serial(port='COM5',
                        baudrate=230400,
                        timeout=5.0,
                        bytesize=8,
                        parity='N',
                        stopbits=1)

    tmpString = ""
    lines = list()
    angles = list()
    distances = list()
    angles_norm = list()
    distances_norm = list()

    i = 0
    while True:
        loopFlag = True
        flag2c = False

        if i % 40 == 39:
            lidarnormdata = LidarNormData(angles_norm, distances_norm, angles, distances)
            return lidarnormdata
            angles.clear()
            distances.clear()
            i = 0

        while loopFlag:
            b = ser.read()
            tmpInt = int.from_bytes(b, 'big')  # Transfer bit to INT

            if tmpInt == 0x54:
                tmpString += b.hex() + " "
                flag2c = True
                continue

            elif tmpInt == 0x2c and flag2c:
                tmpString += b.hex()
                if not len(tmpString[0:-5].replace(' ', '')) == 90:
                    tmpString = ""
                    loopFlag = False
                    flag2c = False
                    continue

                lidarData = CalcLidarData(tmpString[0:-5])

                angles.extend(lidarData.Angle_i)
                distances.extend(lidarData.Distance_i)
                angles_norm.extend(lidarData.Angle_norm)
                distances_norm.extend(lidarData.Distance_norm)

                tmpString = ""
                loopFlag = False
            else:
                tmpString += b.hex() + " "

            flag2c = False

        i += 1
        lidar_time -= 1

    ser.close()
