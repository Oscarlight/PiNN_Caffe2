import numpy as np
import csv

def parse_mdm_to_nparray(file_name):
	'''
	   Read data from .mdm files. Output is two dictionaries: 1. Data 2. Header
	   In Data dict, the keys are in string formats. The data are in float format.
	   In Header dict, everything is in string formats.
	'''
	with open(file_name,'r') as f:

		tmp = ''
		tmp = f.readline().split()

		# Ignore the uesless lines
		if len(tmp) == 0 or tmp[0] =='!':
			tmp = f.readline().split()

		# Read header (including inputs and outputs)
		header = []
		inputs = []
		outputs = []
		if (tmp[0] == "BEGIN_HEADER"):
			tmp = f.readline().split()
			while (tmp[0] != "END_HEADER"):
				header.append(tmp)
				tmp = f.readline().split()

		# Separate inputs and outputs
		i = 0
		if (header[i][0] == 'ICCAP_INPUTS'):
			while (header[i][0] != 'ICCAP_OUTPUTS'):
				i = i + 1
				inputs.append(header[i-1])
		if (header[i][0] == 'ICCAP_OUTPUTS'):
			while (i < len(header)):
				outputs.append(header[i])
				i = i+1

		# Read Data
		Condition = []
		Data_tmp = []

		while (1):
			tmp = f.readline().split()
			# Ignore the uesless lines
			if len(tmp) == 0 or tmp[0] =='!':
				tmp = f.readline().split()
				if len(tmp) == 0:
					break
			cond = []
			data = []
			if (tmp[0] == "BEGIN_DB"):
				tmp = f.readline().split()
				while (tmp[0] == "ICCAP_VAR"):
					cond.append(tmp)
					tmp = f.readline().split()
					if len(tmp) ==0:
						tmp = f.readline().split()
				while (tmp[0] != "END_DB"):
					data.append(tmp)
					tmp = f.readline().split()

			# Transfer string to float
			data[1:] = [map(float,e) for e in data[1:]]

			Condition.append(cond)
			Data_tmp.append(data)

		# Construct Data dictionary
		Data = {}

		cond_val_num = len(Condition[0])
		cond_num = len(Condition)
		data_val_num = len(Data_tmp[0][0])
		data_num = len(Data_tmp[0])-1

		store_tmp = []
		data_val_count = 0
		cond_val_count = 0
		# Data
		while (data_val_count < data_val_num):
			data_store_tmp = []
			cond_count = 0
			while (cond_count < cond_num):
				data_count = 0
				while (data_count < data_num):
					data_store_tmp.append(Data_tmp[cond_count][data_count+1][data_val_count])
					data_count = data_count+1
				cond_count = cond_count+1
			data_val_count = data_val_count+1
			store_tmp.append(data_store_tmp)
		# Condition
		while (cond_val_count < cond_val_num):
			cond_store_tmp = []
        		cond_count = 0
			while (cond_count < cond_num):
				data_count = 0
				while (data_count < data_num):
					cond_store_tmp.append(float(Condition[cond_count][cond_val_count][2]))
					data_count = data_count+1
				cond_count = cond_count+1
			cond_val_count = cond_val_count+1
			store_tmp.append(cond_store_tmp)

		# Combine together
		Data[Data_tmp[0][0][0][1:]] = store_tmp[0]
		count = 1
		while (count < data_val_num):
			Data[Data_tmp[0][0][count]] = store_tmp[count]
			count = count+1
		while (count < data_val_num+cond_val_num):
			Data[Condition[0][count-data_val_num][1]] = store_tmp[count]
			count = count+1

		# Construct Header dictionary
		Header = {}
		Header['Inputs'] = inputs
		Header['Outputs'] = outputs

		return Header, Data

def read_dc_iv_mdm(file_name):
	header, data = parse_mdm_to_nparray(file_name)
	# assert whether is it DC IV data
	assert ('freq' not in data.keys()),'The input data is not dc measurement, abort!'

	if ('Vd' in data.keys()):
		vd = np.array(data['Vd'], dtype = np.float32)
		vg = np.array(data['Vg'], dtype = np.float32)
		ids = np.array(data['Id'], dtype = np.float32)
	elif ('vd' in data.keys()):
		vd = np.array(data['vd'], dtype = np.float32)
		vg = np.array(data['vg'], dtype = np.float32)
		ids = np.array(data['id'], dtype = np.float32)
	else:
		raise Exception('Vd not found!')

	return vg, vd, ids
	
def read_s_par_mdm(file_name):
    header, data = parse_mdm_to_nparray(file_name)

    if ('freq' in data.keys()):
    	freq = np.array(data['freq'],dtype = np.float32)
    elif('#freq' in data.keys()):
    	freq = np.array(data['#freq'],dtype = np.float32)
    else:
    	raise Exception('This is not ac measurement, abort!')

    if ('Vg' in data.keys()):
    	vg = np.array(data['Vg'],dtype = np.float32)
    elif('#Vg' in data.keys()):
    	vg = np.array(data['#Vg'],dtype = np.float32)
    else:
    	raise Exception('Vg not found')

    if ('Vd' in data.keys()):
    	vd = np.array(data['Vd'],dtype = np.float32)
    elif('#Vd' in data.keys()):
    	vd = np.array(data['#Vd'],dtype = np.float32)
    else:
    	raise Exception('Vd not found')

    if ('Id' in data.keys()):
    	id = np.array(data['Id'],dtype = np.float32)
    else:
    	id = "not there"

    s11arr = np.array(data["R:s(1,1)"]) + 1j*np.array(data["I:s(1,1)"])
    s12arr = np.array(data["R:s(1,2)"]) + 1j*np.array(data["I:s(1,2)"])
    s21arr = np.array(data["R:s(2,1)"]) + 1j*np.array(data["I:s(2,1)"])
    s22arr = np.array(data["R:s(2,2)"]) + 1j*np.array(data["I:s(2,2)"])


    return s11arr,s12arr,s21arr,s22arr,freq,vg,vd,id

def read_dc_iv_csv(filename):
    vg = []
    vd = []
    ids = []
    with open(filename) as csvDataFile:
        csvReader = csv.DictReader(csvDataFile,delimiter=',')
        rows = list(csvReader)

        for row in rows:
        	vg.append(row['Vg'])
        	vd.append(row['Vd'])
        	ids.append(row['Id'])
        	
        vg = np.array(map(float, vg), dtype = np.float32)
        vd = np.array(map(float, vd), dtype = np.float32)
        ids = np.array(map(float, ids), dtype = np.float32)
 
    return vg,vd,ids

def read_s_par_csv(filename):
    vg = []
    vd = []
    freq = []
    s11arr = []
    s12arr = []
    s21arr = []
    s22arr = []
    r11 = []
    i11 = []
    r12 = []
    i12 = []
    r21 = []
    i21 = []
    r22 = []
    i22 = []

    with open(filename) as csvDataFile:
        csvReader = csv.DictReader(csvDataFile,delimiter=',')
        rows = list(csvReader)

        for row in rows:
        	vg.append(row['Vg'])
        	vd.append(row['Vd'])
        	freq.append(row['freq'])
        	r11.append(row['R:s(1,1)'])
        	i11.append(row['I:s(1,1)'])
        	r12.append(row['R:s(1,2)'])
        	i12.append(row['I:s(1,2)'])
        	r21.append(row['R:s(2,1)'])
        	i21.append(row['I:s(2,1)'])
        	r22.append(row['R:s(2,2)'])
        	i22.append(row['I:s(2,2)'])

        vg = [float(e) for e in vg]
        vd = [float(e) for e in vd]
        freq = [float(e) for e in freq]
        r11 = [float(e) for e in r11]
        i11 = [float(e) for e in i11]
        r12 = [float(e) for e in r12]
        i12 = [float(e) for e in i12]
        r21 = [float(e) for e in r21]
        i21 = [float(e) for e in i21]
        r22 = [float(e) for e in r22]
        i22 = [float(e) for e in i22]

        s11arr = np.array(r11) + 1j*np.array(i11)
    	s12arr = np.array(r12) + 1j*np.array(i12)
    	s21arr = np.array(r21) + 1j*np.array(i21)
    	s22arr = np.array(r22) + 1j*np.array(i22)
    	id = "not there"
 
    return s11arr,s12arr,s21arr,s22arr,freq,vd,vd,id


if __name__ == '__main__':
	pass
