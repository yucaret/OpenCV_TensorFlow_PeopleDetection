import cv2
import time
import numpy as np
import imutils
import tensorflow as tf

print('OPENCV: ' + cv2.__version__)
print('TENSORFLOW: ' + tf.__version__)

# Video de Escritura
video_writer = None
video_writer_name = 'video_output.avi'

time.sleep(2.0)

# Video de Lectura

CLASSES = ['person']

# Lectura de videos
camara_1 = cv2.VideoCapture('input/camara_1_1.mp4')
 
# Verifica si los videos abrieron
if (camara_1.isOpened()== False): 
  print("Error opening video stream from camara_1")

# Parametros  
people_in = 0
people_out = 0

# Tamaño de Frame Empty
r_move_point =  100
c_move_point =  0

r_empty_frame_varriba = 570
c_empty_frame_varriba = 920

# Angulos de Camara 1
angle_Line_1_3 = (344/180)*np.pi
angle_Line_1_2 = (37.3/180)*np.pi
  
angle_rotate_1 = 2*np.pi - angle_Line_1_3
angle_rotate_2 = angle_rotate_1 + angle_Line_1_2

# Lectura de Modelo Custom
with tf.io.gfile.GFile('tracking_people_model.pb', 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# Abriendo la sesion
with tf.compat.v1.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # Mientras no acaben los videos de la camara 1 y 3
    while(camara_1.isOpened()):
        
        # Capture frame-by-frame
        ret_1, frame_1 = camara_1.read()
        
        if (ret_1 == True):
            
            # Reduccion de tamaño de ca frame
            frame_1 = imutils.resize(frame_1, width=750)
            
            # Copia de los frames
            frame_1_copy = frame_1.copy()
            
            # Conversion de BGR a RGB
            frame_1_rgb = cv2.cvtColor(frame_1,cv2.COLOR_BGR2RGB)
            
            # Creacion del frame vacio Union de Video
            empty_frame_union = np.zeros((336,901,3),dtype='uint8')
            
            # Creacion del frame Vista de Arriba
            empty_frame_varriba = np.zeros((r_empty_frame_varriba, c_empty_frame_varriba,3),dtype='uint8')
            
            cv2.line(empty_frame_varriba, (0, 0), (0, r_empty_frame_varriba), (155, 255, 0), 2)
            cv2.line(empty_frame_varriba, (603, 0), (603, r_empty_frame_varriba), (155, 255, 0), 2)
            
            cv2.line(empty_frame_varriba, (0, 0 + r_move_point), (c_empty_frame_varriba, 0 + r_move_point), (0, 155, 255), 2)
            cv2.line(empty_frame_varriba, (0, 417 + r_move_point), (c_empty_frame_varriba, 417 + r_move_point), (0, 155, 255), 2)
            
            cv2.rectangle(empty_frame_varriba, (0, 0), (empty_frame_varriba.shape[1], empty_frame_varriba.shape[0]), (255, 255, 255), 3)
            
            # Alto (rows - r) y Ancho (columns = c) del Frame 1
            r_frame_1 = frame_1.shape[0]
            c_frame_1 = frame_1.shape[1]
            
            # Puntos de Lineas Fijas para la Camara 1
            x1_Line1_frame_1 = 0 - 548
            y1_Line1_frame_1 = 0
            
            x2_Line1_frame_1 = (c_frame_1 - 548)
            y2_Line1_frame_1 = round(c_frame_1*np.tan(angle_Line_1_2),0).astype('int')
            
            x1_Line2_frame_1 = 0 + 250
            y1_Line2_frame_1 = 0
            
            x2_Line2_frame_1 = (c_frame_1 + 250)
            y2_Line2_frame_1 = round(c_frame_1*np.tan(angle_Line_1_2),0).astype('int')
            
            x1_Line3_frame_1 = 0
            y1_Line3_frame_1 = 0 + 250
            
            x2_Line3_frame_1 = c_frame_1
            y2_Line3_frame_1 = round(c_frame_1*np.tan(angle_Line_1_3),0).astype('int') + 250
            
            x1_Line4_frame_1 = 0
            y1_Line4_frame_1 = 0 + 598
            
            x2_Line4_frame_1 = c_frame_1
            y2_Line4_frame_1 = round(c_frame_1*np.tan(angle_Line_1_3),0).astype('int') + 598

            # Puntos de los cruces de 2 lineas
            x_axis1_frame_1 = (y1_Line3_frame_1 - (-x1_Line1_frame_1*np.tan(angle_Line_1_2))) / (np.tan(angle_Line_1_2) - np.tan(angle_Line_1_3))
            y_axis1_frame_1 = x_axis1_frame_1*np.tan(angle_Line_1_3) + y1_Line3_frame_1
            
            x_axis1_frame_1 = round(x_axis1_frame_1, 0).astype('int')
            y_axis1_frame_1 = round(y_axis1_frame_1, 0).astype('int')
            
            x_axis2_frame_1 = (y1_Line4_frame_1 - (-x1_Line1_frame_1*np.tan(angle_Line_1_2))) / (np.tan(angle_Line_1_2) - np.tan(angle_Line_1_3))
            y_axis2_frame_1 = x_axis2_frame_1*np.tan(angle_Line_1_3) + y1_Line4_frame_1
            
            x_axis2_frame_1 = round(x_axis2_frame_1, 0).astype('int')
            y_axis2_frame_1 = round(y_axis2_frame_1, 0).astype('int')
            
            x_axis3_frame_1 = (y1_Line3_frame_1 - (-x1_Line2_frame_1*np.tan(angle_Line_1_2))) / (np.tan(angle_Line_1_2) - np.tan(angle_Line_1_3))
            y_axis3_frame_1 = x_axis3_frame_1*np.tan(angle_Line_1_3) + y1_Line3_frame_1
            
            x_axis3_frame_1 = round(x_axis3_frame_1, 0).astype('int')
            y_axis3_frame_1 = round(y_axis3_frame_1, 0).astype('int')
            
            
            radio_1 = np.sqrt(pow((x_axis1_frame_1 - x_axis2_frame_1), 2) +
                              pow((y_axis1_frame_1 - y_axis2_frame_1), 2))
            
            radio_2 = np.sqrt(pow((x_axis1_frame_1 - x_axis3_frame_1), 2) +
                              pow((y_axis1_frame_1 - y_axis3_frame_1), 2))
            
            radio_1 = round(radio_1, 0).astype('int')
            
            radio_2 = round(radio_2, 0).astype('int')

            # Ejecucion de modelos
            out_frame_1 = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                                    sess.graph.get_tensor_by_name('detection_scores:0'),
                                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                                    sess.graph.get_tensor_by_name('detection_classes:0')],
                                   feed_dict={'image_tensor:0': frame_1_rgb.reshape(1,
                                                                                    frame_1_rgb.shape[0],
                                                                                    frame_1_rgb.shape[1], 3)})
            
            
            # Visualize detected bounding boxes
            num_detections = int(out_frame_1[0][0])
            
            # Cantidad de Personas Fuera y Dentro
            people_in = 0
            people_out = 0
            
            for i in range(num_detections):
                classId = int(out_frame_1[3][0][i])
                score = float(out_frame_1[1][0][i])
                box = [float(v) for v in out_frame_1[2][0][i]]
                
                if score > 0.99:
                    
                    #print('classId: ' + str(classId) + ' score: ' + str(score))
                    
                    x1 = int(box[1] * c_frame_1)
                    y1 = int(box[0] * r_frame_1)
                    
                    x2 = int(box[3] * c_frame_1)
                    y2 = int(box[2] * r_frame_1)
                    
                    
                    midx = int(round((x1 + x2)/2,0))
                    midy = int(round((y1 + y2)/2,0))
                    
                    midx_1 = midx - x_axis1_frame_1
                    
                    midy_1 = midy - y_axis1_frame_1
            
                    radio_3 = np.sqrt(pow(midx_1,2) + pow(midy_1,2))
                    
                    pendiente = (midy_1) / (midx_1)
                    
                    angle_betha = np.arctan(pendiente)
                    
                    angle_betha_2 = angle_betha + angle_rotate_1
                    
                    midx_2 = radio_3*np.cos(angle_betha_2)
            
                    midy_2 = radio_3*np.sin(angle_betha_2)
                    
                    midx_3 = radio_3*np.cos(angle_betha_2) - radio_3*np.sin(angle_betha_2)/np.tan(angle_rotate_2)
                    
                    midy_3 = radio_3*np.sin(angle_betha_2)/np.sin(angle_rotate_2)
                    
                    radio_3 = np.sqrt(pow(x_axis1_frame_1 - midx, 2) + pow(y_axis1_frame_1 - midy, 2))
            
                    midx_3 = int(round(midx_3, 0)) + c_move_point
                    midy_3 = int(round(midy_3, 0)) + r_move_point
                    
                    if (midx_3 < 603):
                        people_in = people_in + 1
                    else:
                        people_out = people_out + 1
                    
                    radio_3 = int(round(radio_3, 0))
                    
                    cv2.rectangle(frame_1, (x1, y1), (x2, y2), (128, 6, 152), thickness=2)
                    
                    cv2.circle(frame_1, (midx, midy), 3,  (255, 0, 0), 2)
                    
                    cv2.rectangle(empty_frame_varriba, (midx_3 - 25, midy_3 - 25), (midx_3 + 25, midy_3 + 25), (128, 6, 152), thickness=2)
                    
                    cv2.circle(empty_frame_varriba, (midx_3, midy_3), 3, (255, 0, 0), 2)
            
            # Dibujando Lineas de area donde se realiza el tracking
            cv2.line(frame_1, (x1_Line1_frame_1, y1_Line1_frame_1), (x2_Line1_frame_1, y2_Line1_frame_1), (155, 255, 0), 2)
            
            cv2.line(frame_1, (x1_Line2_frame_1, y1_Line2_frame_1), (x2_Line2_frame_1, y2_Line2_frame_1), (155, 255, 0), 2)
            
            cv2.line(frame_1, (x1_Line3_frame_1, y1_Line3_frame_1), (x2_Line3_frame_1, y2_Line3_frame_1), (0, 155, 255), 2)
            
            cv2.line(frame_1, (x1_Line4_frame_1, y1_Line4_frame_1), (x2_Line4_frame_1, y2_Line4_frame_1), (0, 155, 255), 2)
            
            # Reduciendo de Tamaño los Frames
            frame_1 = imutils.resize(frame_1, width = 600)
            
            empty_frame_varriba = imutils.resize(empty_frame_varriba, width = 300)
            
            # Uniendo los Frames
            empty_frame_union[0: frame_1.shape[0], 0: frame_1.shape[1]] = frame_1[0 : frame_1.shape[0], 0: frame_1.shape[1]]
            empty_frame_union[0 + 75: empty_frame_varriba.shape[0] + 75, frame_1.shape[1]: frame_1.shape[1] + empty_frame_varriba.shape[1]] = empty_frame_varriba[0: empty_frame_varriba.shape[0], 0: empty_frame_varriba.shape[1]]
            
            cv2.putText(empty_frame_union, '** Aerial view **', ( 680, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(empty_frame_union, 'People In: ' + str(people_in), ( 630, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            cv2.putText(empty_frame_union, 'People Out: ' + str(people_out), ( 800, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            cv2.putText(empty_frame_union, 'jorge Eduardo Vicente Hernandez', ( 350, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 0, 0), 2)
            
            # Empezando a grabar el frame del video en la ruta especificada
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                video_writer = cv2.VideoWriter('output/' + video_writer_name, fourcc, 20, (empty_frame_union.shape[1], empty_frame_union.shape[0]), True)

            # Capturar la imagen del frame en el video
            if video_writer is not None:
                video_writer.write(empty_frame_union)
            
            # Mostrando los frames
            cv2.imshow('Traking',empty_frame_union)
            
            # Presionar 'q' para salir
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
           
        # Salimos del Loop
        else:
            break
 
# Cuando todo este listo, despliega el video
camara_1.release()
 
# Cierran todos los frames
cv2.destroyAllWindows()

# Guardamos el video en la ruta
if video_writer is not None:
    video_writer.release()