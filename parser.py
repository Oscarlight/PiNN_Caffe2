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
		    print('header',header)

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
   		  print('inputs:', inputs)
    		  print('outputs:', outputs)

   		  # Read Data
   		  Condition = []
   		  Data = []

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
       		  print('cond:', cond)

       		  # Transfer string to float
       		  j = 1
        	  while (j < len(data)):
           		k = 0
            		while (k < len(data[0])):
               			data[j][k] = float(data[j][k])
                		k = k+1
            		j = j+1
        	  print('data:', data)

        	  Condition.append(cond)
        	  Data.append(data)

    		  print('Condition:',Condition)
    		  print('Data:',Data)
