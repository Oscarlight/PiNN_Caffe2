# @ Xiang Li: write a python scirpt to parse the data in .mdm file from 
# HEMT_Bo folder to numpy arrays of features and labels

def parse_mdm_to_nparray(file_name):

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
		count = 0
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
