# Master_CSI_Auth

# 'Secure_connection' folder includes the secure socket connection programs with the CSI reader, processor and authenticator.

1. 'server_new.py' is the server program running on AX200 mini PC, responsible for decrypting and receiving the message, also reading, processing, authenticate the CSI. 
'server_new_multiple.py' is an example program for the multiple user access situation.
2. 'client_new.py' is the client program running on client machine, responsible for encrypting and sending message to the server. 
'client_new_multiple.py' is an example program for the  multiple user access situation.
3. 'csi_anlysis_new.py' defines the CSI processing and authentication functions.
4. 'secure_connection.py' defines the encryption and decryption functions in the connection.
5. '*.sh' files are the shell scripts used to read and update test or user profile file from the system.
6. 'csi_analysis,ipynb' is used to visualize the results. The data and results are saved in other subfolders.

# 'CSI_authentication' folder includes the basic CSI processing and authentication algorithms.

1. 'csikit.ipynb' includes the basic data processing functions, also with the data processing results visualization.
2. 'clustering.ipynb' includes the basic authentication functions, also with the accuracy results visualization.
(The processed data is saved in other subfolders)

# 'matlab' folder includes some original pcap files used in initial feasibility study in 'CSI_authentication' folder.
