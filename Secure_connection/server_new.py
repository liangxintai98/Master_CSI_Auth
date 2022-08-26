from cgi import test
import socket 
from datetime import datetime

from matplotlib.pyplot import axis
from requests import session
#from rsa import decrypt
import secure_connection
from cryptography.hazmat.backends import default_backend  
from cryptography.hazmat.primitives.asymmetric import padding  
from cryptography.hazmat.primitives import hashes  
from cryptography.hazmat.primitives.serialization import load_pem_private_key  
from cryptography.hazmat.primitives.serialization import load_pem_public_key  
import copy
import time
import csi_analysis
from os.path import exists
import subprocess
import statistics
import os
import threading
import tracemalloc
import psutil
import numpy as np
import hmac
import csi_analysis_new
#from Crypto.Cipher import AES
#from Crypto.Util.Padding import pad, unpad
from base64 import b64encode, b64decode
from binascii import unhexlify


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

tracemalloc.start()

def get_input():
    print()
    data = input
    return data

def main():
    
    p = psutil.Process(os.getpid())
    cpu_rec = []
    # print(procc_id)
    input_thread = threading.Thread(target=get_input)
    # HOST = '192.168.1.149'  # Standard loopback interface address (localhost)
    HOST = '192.168.3.1'
    # HOST = '127.0.0.1'
    PORT = 8889     # Port to listen on (non-privileged ports are > 1023)

    auth_method = 'digital'
    # auth_method = 'hmac'

    #here we get server private key and client public key
    serverPvKey = load_pem_private_key(open('rsakey.pem', 'rb').read(),b"12345",default_backend())  
    clientPubKey = load_pem_public_key(open('rsapub_client_2048.pem', 'rb').read(),default_backend())
    iv = "7bde5a0f3f39fd658efc45de143cbc94"
    iv = unhexlify(iv)
    csi_data = np.empty((50,108))

    # s_key = b""
    # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        
        s.bind((HOST, PORT))
        # s.listen()
        # conn, addr = s.accept()
        session_key = b""
        # with conn:

        flag, addr = s.recvfrom(1024)
        print('Connected by', addr)
        # flag = conn.recv(1024)
        
        # print(data)
        #If the client wants to send new session key we accept it
        if flag == b"NEW_KEY":
            # ciphered_session_key = conn.recv(1024)
            ciphered_session_key = s.recv(1024)
            print("ciphered_session_key received and = :", ciphered_session_key)
            session_key = secure_connection.decrypt_session_key(ciphered_session_key, serverPvKey)
            # cipher = AES.new(session_key, AES.MODE_CBC, iv)
        if flag == b"FILE":
            pass
        if flag == b"TEXT":
            pass
            # print(flag)
        # s_key += data
        # f.write(data)
        # counter = 1

        n = 0
        i = 0
        k = 0
        count = 0
        csi_data_test = np.empty((50,108))
        csi_data_train = np.empty((50,108))
        # csi_data = np.empty((50,108))

        while flag:
            
            # start = time.time()
            file_exist_profile = exists('/home/nss/profile.dat')
            # file_exist_profile = exists('/Users/liangxintai/Desktop/live_pcap/iphone12pro_lab_s1.pcap')

            if file_exist_profile == True:
                
                df_train, matrix = csi_analysis_new.csi_analyzer(path= '/home/nss/profile.dat',num_frame= 1500, flag='train')
                # df_train, matrix = csi_analysis.csi_analyzer(csi_file_name='iphone12pro_lab_s1.pcap')
                # all = np.append(csi_data_train, matrix, axis=0)
                
                if k == 0:
                    all = np.append(csi_data, matrix, axis=0)
                    np.savetxt('csi_data_all.csv', all[50:,:], delimiter=',')
                else:
                    all = np.append(all, matrix, axis = 0)
                    np.savetxt('csi_data_all.csv', all[50:,:], delimiter=',')
                # print(df_train)
                var = csi_analysis_new.var_cal(df_train)

                if var > 0.8:
                    x_train = csi_analysis_new.train_process(df_train)
                    # df_test = csi_analysis.csi_analyzer(csi_file_name='live_test.pcap')
                    # x_test = csi_analysis.test_process(df_test)
                    authenticator = csi_analysis_new.gen_authenticator(x_train)
                    initial = '0'
                    check = '1'
                else:
                    print('********* Profile not stable *********')
                    os.remove('/home/nss/profile.dat')
                    initial = '0'
                    check = '0'
                    subprocess.Popen(['sh', './csi_cap_profile.sh'])
                
                
            elif file_exist_profile == False:
                initial = '0'
                check = '0'
                subprocess.Popen(['sh', './csi_cap_profile.sh'])

            while flag:

                start = time.time()
                
                file_exist_profile = exists('/home/nss/profile.dat')
                # file_exist_profile = exists('/Users/liangxintai/Desktop/live_pcap/iphone12pro_lab_s1.pcap')

                if file_exist_profile == True and initial == '0':

                    if check == '1':

                        file_exist_test = exists('/home/nss/test.dat')

                        if file_exist_test == True:
                            os.remove('/home/nss/test.dat')
                            result = b'0'
                            test_cap = subprocess.Popen(['sh', './csi_cap_test.sh'])
                            print('Test_file capturing ...')
                        elif file_exist_test == False:
                            result = b'0'
                            test_cap = subprocess.Popen(['sh', './csi_cap_test.sh'])
                            print('Test_file capturing ...')
                        initial = '1'

                    elif check == '0':

                        df_train, matrix = csi_analysis_new.csi_analyzer(path = '/home/nss/profile.dat', num_frame=1500, flag='train')
                        # all = np.append(csi_data_train, matrix, axis=0)

                        if k == 0:
                            all = np.append(csi_data, matrix, axis=0)
                            np.savetxt('csi_data_all.csv', all[50:,:], delimiter=',')
                        else:
                            all = np.append(all, matrix, axis = 0)
                            np.savetxt('csi_data_all.csv', all[50:,:], delimiter=',')

                        var = csi_analysis_new.var_cal(df_train)
                        
                        file_exist_test = exists('/home/nss/test.dat')
                        
                        if file_exist_test == True:
                            os.remove('/home/nss/test.dat')
                            result = b'0'
                            test_cap = subprocess.Popen(['sh', './csi_cap_test.sh'])
                            print('Test_file capturing ...')
                        elif file_exist_test == False:
                            result = b'0'
                            test_cap = subprocess.Popen(['sh', './csi_cap_test.sh'])
                            print('Test_file capturing ...')

                        if var > 0.8:
                            x_train = csi_analysis_new.train_process(df_train)
                            authenticator = csi_analysis_new.gen_authenticator(x_train)
                            initial = '1'

                        else:
                            print('********* Profile not stable *********\n')
                            os.remove('/home/nss/profile.dat')
                            exist = exists('/home/nss/test.dat')
                            if exist == True:
                                os.remove('/home/nss/test.dat')
                            test_cap.terminate()
                            k=k+1
                            break
                    
                elif file_exist_profile == True and initial == '1':
                    file_exist_test = exists('/home/nss/test.dat')
                    if file_exist_test == True:
                        if i%50 == 0:
                            i = 0
                            perc = count / 50
                            count = 0

                            # df_test, matrix = csi_analysis_new.csi_analyzer(path = '/home/nss/test.dat', num_frame= 100, flag='test')
                            df_test, matrix = csi_analysis_new.csi_analyzer(path = '/var/log/csi.dat', num_frame = 60, flag='test')
                            subprocess.Popen(['sh', './clean.sh'])
                            # df_test, matrix = csi_analysis.csi_analyzer(csi_file_name='iphone12pro_lab_s1.pcap')
                            
                            # csi_data_test = np.append(csi_data_test, matrix[0:50,:], axis=0)
                            all = np.append(all, matrix, axis=0)
                            # np.savetxt('csi_data_test.csv', csi_data_test[50:,:], delimiter=',')
                            np.savetxt('csi_data_all.csv', all[50:,:], delimiter=',')

                            x_test = csi_analysis_new.test_process(df_test)
                        print('Using test file ...')
                        result = csi_analysis_new.authenticate(authenticator, x_test[i,:])
                        if result == b'0':
                            count = count + 1
                        i = i+1
                    elif file_exist_test == False:
                        result = b'0'
                        # subprocess.Popen(['sh', './test_cap.sh'])
                        print('Test_file capturing ...')
                    initial = '2'
                
                elif file_exist_profile == True and initial == '2':
                    file_exist_test = exists('/home/nss/test.dat')
                    # file_exist_test = exists('/Users/liangxintai/Desktop/live_pcap/iphone12pro_lab_s1.pcap')
                    if file_exist_test == True:
                        if i%50 == 0:
                            i = 0
                            perc = count / 50
                            count = 0
                            
                            if perc > 0.8:
                                print('******* Fluctation detected *******\n')
                                
                                #os.remove('/home/nss/profile.dat')
                                #os.remove('/home/nss/test.dat')
                                #test_cap.terminate()
                                k=k+1
                                #break

                            # df_test, matrix = csi_analysis_new.csi_analyzer(path = '/home/nss/test.dat', num_frame= 100, flag='test')
                            df_test, matrix = csi_analysis_new.csi_analyzer(path = '/var/log/csi.dat', num_frame = 60, flag='test')
                            subprocess.Popen(['sh', './clean.sh'])
                            # df_test, matrix = csi_analysis.csi_analyzer(csi_file_name='iphone12pro_lab_s1.pcap')
                            # csi_data_test = np.append(csi_data_test, matrix[0:50,:], axis=0)
                            all = np.append(all, matrix, axis=0)
                            # np.savetxt('csi_data_test.csv', csi_data_test[50:,:], delimiter=',')
                            np.savetxt('csi_data_all.csv', all[50:,:], delimiter=',')

                            x_test = csi_analysis_new.test_process(df_test)
                        print('Using test file ...')
                        result = csi_analysis_new.authenticate(authenticator, x_test[i,:])
                        if result == b'0':
                            count = count + 1
                        i = i+1
                    elif file_exist_test == False:
                        result = b'0'
                        # subprocess.Popen(['sh', './test_cap.sh'])
                        print('Test_file capturing ...')

                    us = p.cpu_percent()
                    # p = psutil.Process(os.getpid())
                    # print('*****************')
                    # print(us)
                    # print('*****************\n')
                    
                    cpu_rec.append(us)   # recording CPU usage

                    # f = open("cpu_server_time.txt", "w")
                    # for d in cpu_rec:
                    #     f.write(f"{d}\n")
                    # f.close()
                    
                elif file_exist_profile == False:
                    result = b'0'
                    print('User_profile buidling...')
                    
                    ''' 
                if n >=0 and n < 400:
                    result = b'0'
                elif n>=400 and n < 900: 
                    result = b'1'
                elif n >=900 and n < 1200:
                    result = b'0'
                elif n >= 1200 and n < 2000:
                    result = b'1'
                elif n >= 2000:
                    result = b'0'
                    print('okokokkokkokokokokokokokokoookokokokookokkokkko')
                    
                    '''
                
                n = n+1
                # conn.send(result)
                
                s.sendto(result,addr)

                # time.sleep(1)

                flag, ciphered_text, signature = [i for i in s.recv(1024).split(b'$$$$$')]
                
                print('**************************************')
                # print(flag)
                print(ciphered_text)
                print(signature)
                out = b64encode(ciphered_text).decode('utf-8')

                if flag == b"TEXT":

                    plain_text = secure_connection.decrypt_message(ciphered_text, session_key)
                    
                    # plain_text = cipher.decrypt(ciphered_text)
                    # plain_text = unpad(cipher.decrypt(b64decode(out)), AES.block_size).decode('utf-8')

                    print("The text sent from client is:",plain_text.decode())

                    if result == b'1':
                        print('CSI authentication SUCCESS')
                        print("This message is valid")
                        print('**************************************\n')
                        text = 'SERVER: Message receive SUCCESS'
                        text = text.encode()
                        ciphered_text = secure_connection.encrypt_message(text, session_key)
                        # ciphered_text = cipher.encrypt(text)
                        signature = b'no_signature'

                        
                    elif result == b'0':
                        print('CSI authentication FAIL')
                        print("Using SIGNATURE authentication")
                        print('......')
                        
                        if auth_method == 'digital':
                            if secure_connection.signature_is_valid(plain_text, signature, clientPubKey):
                                print("SIGNATURE authentication SUCCESS")
                                print("This message is valid")
                                print('**************************************\n')
                                text = 'SERVER: Message receive SUCCESS'
                                text = text.encode()
                                ciphered_text = secure_connection.encrypt_message(text, session_key)
                                # ciphered_text = cipher.encrypt(text)
                                signature_n = secure_connection.sign_message(text, serverPvKey)
                            else:
                                print("SIGNATURE authentication FAIL")
                                print("This message is invalid")
                                print('**************************************\n')
                                text = 'SERVER: Message receive FAIL'
                                text = text.encode()

                                ciphered_text = secure_connection.encrypt_message(text, session_key)
                                # ciphered_text = cipher.encrypt(text)
                                signature_n = secure_connection.sign_message(text, serverPvKey)

                        elif auth_method == 'hmac':
                            # print(session_key)
                            signature_here = secure_connection.hmac_gen(plain_text, session_key)
                            if signature_here == signature.decode():
                                print("SIGNATURE authentication SUCCESS")
                                print("This message is valid")
                                print('**************************************\n')
                                text = 'SERVER: Message receive SUCCESS'
                                text = text.encode()
                                ciphered_text = secure_connection.encrypt_message(text, session_key)
                                # ciphered_text = cipher.encrypt(text)
                                signature_n = secure_connection.sign_message(text, serverPvKey)
                            else:
                                print("SIGNATURE authentication FAIL")
                                print("This message is invalid")
                                print('**************************************\n')
                                text = 'SERVER: Message receive FAIL'
                                text = text.encode()
                                ciphered_text = secure_connection.encrypt_message(text, session_key)
                                # ciphered_text = cipher.encrypt(text)
                                signature_n = secure_connection.sign_message(text, serverPvKey)
                        
                    # conn.send(b'$$$$$'.join([ciphered_text, signature_n]))
                    s.sendto(b'$$$$$'.join([ciphered_text, signature_n]), addr)

                    current, peak = tracemalloc.get_traced_memory()
                    # print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

                    # us = p.cpu_percent()
                    # # p = psutil.Process(os.getpid())
                    # print('*****************')
                    # print(us)
                    # print('*****************\n')

                    # cpu_rec.append(us)

                    # f = open("cpu_server_time.txt", "w")
                    # for d in cpu_rec:
                    #     f.write(f"{d}\n")
                    # f.close()

                    # time.sleep(0.1)
                
                end = time.time()
                # print(end-start)
                
                time.sleep(0.05)    

if __name__ == "__main__":
    
    main() 

tracemalloc.stop()

