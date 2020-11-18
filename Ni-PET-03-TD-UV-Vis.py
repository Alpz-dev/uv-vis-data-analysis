from data import *


temps = ["290", "250", "210", "170", "130", "080"]

datas = []

for temp in temps:
    data_tmp = import_data("test data/11-4-2020_Ni-PET_PTLC03_thin_film_" + temp + "K.txt")
    delta1 = data_tmp.y[722 - 220] - data_tmp.y[719 - 220]
    delta2 = data_tmp.y[902 - 220] - data_tmp.y[898 - 220]


    data_new = data_tmp.translate(delta1, start_x = 200, end_x = 719)
    data_new = data_new.translate(delta2, start_x = 200, end_x = 899)
    data_new.y[901 - 220] = data_new.y[902 - 220]
    data_new.y[900 -220] = data_new.y[902 - 220]
    data_new.y[720 - 220] = data_new.y[722 - 220]
    data_new.y[721 - 220] = data_new.y[722 - 220]

    #data_new.y[720 - 220] = data_new.y[721 - 220]
    datas.append(data_new)
    #data_new.plot(1)
    data_new.savgol_filter(15, 2).plot(1, start_x = 300, end_x = 900)
    #data_tmp.plot(1)
    #data_tmp.plot(1)



plt.xlabel("Wavelength (nm)")
plt.ylabel("Abs")
plt.legend(["293K", "240K", "180K", "120K", "80K"])
plt.show()
