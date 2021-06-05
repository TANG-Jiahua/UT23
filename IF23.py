####################################
#   FU Zonghang and Tang jiahua    #
#                                  #
#             IF23                 #
#             UTT                  #
#             2021                 #
#                                  #
####################################
import subprocess
import numpy as np 
import sys
from sklearn import svm
from sklearn.metrics import accuracy_score


postes = ['','','','','']




def mode_scan(zone):
     print("Capture de la zone "+ zone + " démarrée")

     for _ in range(10):

          cmd = "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport -s | sed 's/^[ \t]*//' | cut -d ' ' -f2,3"
          process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)

          output, error = process.communicate()
          output = output.decode().split('\n')
          output = output[:-1]
          

          with open("./dataRSSI.csv", "a") as RSSI_file: 
               RSSI_file.write(zone + ",")
               for data in output:
                    data = data.split(" ")
                    if data[0].lower() == "00:27:e3:07:b3:51":
                         postes[0] = data[1]
                    elif data[0].lower() == "00:27:e3:07:b3:52":
                         postes[1] = data[1]
                    elif data[0].lower() == "00:27:e3:07:b3:55":
                         postes[2] = data[1]
                    elif data[0].lower() == "00:27:e3:07:b3:5e":
                         postes[3] = data[1]
                    elif data[0].lower() == "00:a3:8e:c7:a4:f5":
                         postes[4] = data[1]


               print(postes)
               RSSI_file.write(postes[0]+","+postes[1]+","+postes[2]+","+postes[3]+","+postes[4]+"," + "\n")
     
     print("Capture de la zone "+ zone + " terminée")





def mode_apprentissage(postes):
     with open("./IF23.csv", "r") as RSSI_file:
          zone = []
          rssiTab = [] 
          for line in RSSI_file:
               tmp = line.split(",")
               zone.append(''.join(tmp[0]))
               rssiTab.append(tmp[1:6]) 
         
          X = rssiTab
          Y = zone
          clf = svm.SVC(decision_function_shape='ovo',kernel='poly')
          clf.fit(X, Y)

          prepoly = clf.predict([postes])
          print("Selon la méthode poly, l'emplacement devrait être",prepoly)

          clf = svm.SVC(decision_function_shape='ovo',kernel='rbf')
          clf.fit(X, Y)

          prerbf = clf.predict([postes])
          print("Selon la méthode rbf, l'emplacement devrait être",prerbf)

          clf = svm.SVC(decision_function_shape='ovo',kernel='linear')
          clf.fit(X, Y)

          prelinear = clf.predict([postes])
          print("Selon la méthode linear, l'emplacement devrait être",prelinear)

          clf = svm.SVC(decision_function_shape='ovo',kernel='sigmoid')
          clf.fit(X, Y)

          presigmoid = clf.predict([postes])
          print("Selon la méthode sigmoid, l'emplacement devrait être",presigmoid)


          maxt=mode_test()
          if (maxt == 0):
               pre=prepoly
          elif (maxt == 1):
               pre=prerbf
          elif (maxt == 2):
               pre=prelinear
          elif (maxt == 3):
               pre=presigmoid

          #print(maxt)
          print("La zone prédit par le meilleur modèle est",pre)








def mode_live():
     cmd = "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport -s | sed 's/^[ \t]*//' | cut -d ' ' -f2,3"
     process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)

     output, error = process.communicate()
     output = output.decode().split('\n')
     output = output[:-1]
     
          
     for data in output:
          data = data.split(" ")
          if   data[0].lower() == "00:27:e3:07:b3:51":
               postes[0] = data[1]
          elif data[0].lower() == "00:27:e3:07:b3:52":
               postes[1] = data[1]
          elif data[0].lower() == "00:27:e3:07:b3:55":
               postes[2] = data[1]
          elif data[0].lower() == "00:27:e3:07:b3:5e":
               postes[3] = data[1]
          elif data[0].lower() == "00:a3:8e:c7:a4:f5":
               postes[4] = data[1]

          
          print(postes)
          mode_apprentissage(postes)






def mode_test():
      taux=[0,0,0,0]
      with open("./IF23.csv", "r") as RSSI_file:
          
          zone, rssiTab = transformation_zone(RSSI_file)
         
          X = np.array(rssiTab)
       
          Y = np.array(zone)
         
          clf_p = svm.SVC(decision_function_shape='ovo', kernel='poly')
          clf_p.fit(X, Y)
          

          clf_r = svm.SVC(decision_function_shape='ovo', kernel='rbf')
          clf_r.fit(X, Y)


          clf_l = svm.SVC(decision_function_shape='ovo', kernel='linear')
          clf_l.fit(X, Y)


          clf_s = svm.SVC(decision_function_shape='ovo', kernel='sigmoid')
          clf_s.fit(X, Y)



          with open("./IF23T.csv", "r") as RSSI_file_test:
               
               zone_test, rssiTab_test = transformation_zone(RSSI_file_test)

               X_test = np.array(rssiTab_test)
               
               Y_test = np.array(zone_test)
               pre_p = []
               pre_r = []
               pre_l = []
               pre_s = []


               for unitPre in X:
                    pre_p.append(clf_p.predict([unitPre]))

               taux_poly=accuracy_score(Y, pre_p)
               taux[0]=taux_poly 
          

               for unitPre in X:
                    pre_r.append(clf_r.predict([unitPre]))

               taux_rbf=accuracy_score(Y, pre_r)
               taux[1]=taux_rbf

               for unitPre in X:
                    pre_l.append(clf_l.predict([unitPre]))

               taux_linear=accuracy_score(Y, pre_l)
               taux[2]=taux_linear 

               for unitPre in X:
                    pre_s.append(clf_s.predict([unitPre]))


               taux_sigmoid=accuracy_score(Y, pre_s)
               taux[3]=taux_poly           

               print("Le taux de réussite sur les données d'apprentissage poly est de", taux_poly)
               print("Le taux de réussite sur les données d'apprentissage rbf est de", taux_rbf)
               print("Le taux de réussite sur les données d'apprentissage linear est de", taux_linear)
               print("Le taux de réussite sur les données d'apprentissage sigmoid est de", taux_sigmoid)  



          maxt=taux.index(max(taux))
          if (maxt == 0):
               method="poly"
          elif (maxt == 1):
               method="rbf"
          elif (maxt == 2):
               method="linear"
          elif (maxt == 3):
               method="sigmoid"

          print("Le taux maximum de réussite sur les données d'apprentissage est", method)

          return maxt




def transformation_zone(rssi_file):
     zone_test = []
     rssiTab_test = []

     for line in rssi_file:
          tmp = line.split(",")
          zone_test.append(''.join(tmp[0]))
          rssiTab_test.append(tmp[1:6])

     return zone_test, rssiTab_test










if len(sys.argv) < 2:
     print('-s mode_scan -r test apprentissage -l mode_live -t mode_test')
     exit(1)
mode = sys.argv[1]
if mode == "-s":
     if len(sys.argv) < 3:
          print('-s name zone')
          exit(1)
     if sys.platform=='darwin':
          print('Systeme is mac')
          zone= sys.argv[2]   
          mode_scan(zone)
     else:
          print('Systeme must use MacOS')


elif mode == "-r":
     if len(sys.argv) < 6:
          print('please entre 5 RSSI')
          exit(1)
     postes[0]=sys.argv[2]
     postes[1]=sys.argv[3]
     postes[2]=sys.argv[4]
     postes[3]=sys.argv[5]
     postes[4]=sys.argv[6]
     mode_apprentissage(postes)
elif mode == "-l":

     if sys.platform=='darwin':
          print('Systeme is mac')
          mode_live()
     else:
          print('Systeme must use MacOS')

          
elif mode == "-t":
     mode_test()
else:
     print("-s mode_scan -r test apprentissage -l mode_live -t mode_test")
     exit(1)  













