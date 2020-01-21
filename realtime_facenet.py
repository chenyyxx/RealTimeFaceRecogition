from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
from align import detect_face
import os
import time
import pickle

import face_annoy

print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './align')

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160

        datadir = './data/pre_img'
        dataset = facenet.get_dataset(datadir)
        HumanNames = [cls.name.replace('_', ' ') for cls in dataset][0:-1]
        # HumanNames = ['Akshay Kumar', 'Nawazuddin Siddiqui', 'Salman Khan', 'Shahrukh Khan', 'Sunil Shetty', 'Sunny Deol']    #train human name

        print('Loading feature extraction model')
        modeldir = './model/20180402-114759.pb'
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        # classifier_filename = './classifier/my_classifier1.pkl'
        # classifier_filename_exp = os.path.expanduser(classifier_filename)
        # with open(classifier_filename_exp, 'rb') as infile:
        #     (model, class_names) = pickle.load(infile)
        #     print('load classifier file-> %s' % classifier_filename_exp)

        video_capture = cv2.VideoCapture(0)
        c = 0

        # #video writer
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # out = cv2.VideoWriter('3F_0726.avi', fourcc, fps=30, frameSize=(640,480))

        print('Start Recognition!')
        prevTime = 0
        while True:
            ret, frame = video_capture.read()

            frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)  # resize frame (optional)

            curTime = time.time()  # calc fps
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                print('Detected_FaceNum: %d' % nrof_faces)

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces, 4), dtype=np.int32)

                    for i in range(nrof_faces):
                        # print(i)
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]
                        print(bb)

                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('face is inner of range!')
                            continue

                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped[i] = facenet.flip(cropped[i], False)
                        # print(len(cropped))
                        scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                               interpolation=cv2.INTER_CUBIC)
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))

                        # print(len(scaled))

                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        # print(feed_dict)
                        # get embedded array for the real time detect face

                        # plot the box around the face
                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)  # boxing face

                        # TODO: and then query instead of predicting by model
                        # print(emb_array)
                        annoy = face_annoy.face_annoy()
                        best_class_labels = annoy.query_vector(emb_array[0])
                        print(best_class_labels)
                        name_index = best_class_labels[0]
                        print(HumanNames[name_index[0]])


                        # plot result idx under box
                        text_x = bb[i][0]
                        text_y = bb[i][3] + 20
                        cv2.putText(frame, HumanNames[name_index[0]], (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (0, 0, 255), thickness=1, lineType=2)


                        # This used svc to predict the classification of face
                        # predictions = model.predict_proba(emb_array)
                        # best_class_indices = np.argmax(predictions, axis=1)
                        # best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        #
                        # #plot result idx under box
                        # text_x = bb[i][0]
                        # text_y = bb[i][3] + 20
                        # # print('result: ', best_class_indices[0])
                        # for H_i in HumanNames:
                        #     if HumanNames[best_class_indices[0]] == H_i:
                        #         score = '{0:.2f}'.format(best_class_probabilities[0])
                        #         # if score < 0.5:
                        #         #     continue
                        #         # else:
                        #         result_names = HumanNames[best_class_indices[0]]+' '+str(score)
                        #
                        #         cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        #                     1, (0, 0, 255), thickness=1, lineType=2)
                else:
                    print('Unable to align')

            sec = curTime - prevTime
            prevTime = curTime
            fps = 1 / (sec)
            string = 'FPS: %2.3f' % fps
            text_fps_x = len(frame[0]) - 150
            text_fps_y = 20
            cv2.putText(frame, string, (text_fps_x, text_fps_y),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
            # c+=1
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        # #video writer
        # out.release()
        cv2.destroyAllWindows()
