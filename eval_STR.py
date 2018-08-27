import os, glob, sys, time
import numpy as np
import scipy.io as io
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score
from utils import *
import tensorflow as tf
from yolo_models import build_yolo_v2

build_phoc = import_cphoc()

if __name__ == "__main__":

    img_shape    = (608, 608, 3)
    num_priors   = 13
    phoc_size    = 604
    weights_path = './ckpt/yolo-phoc_175800.ckpt'
    thresh       = 0.002
    n_neighbors  = 10000
    gt_data_path = './datasets/IIIT_STR_V1.0/data.mat' # uses GT format of IIIT_STR & Sports10K datasets
    inp_path     = './datasets/IIIT_STR_V1.0/imgDatabase/'

    # build the model
    model_input  = tf.placeholder(tf.float32, shape=(None,)+img_shape)
    model_output = build_yolo_v2(model_input, num_priors, phoc_size)

    print_ok('In  shape: '+str(model_input.get_shape())+'\n')
    print_ok('Out shape: '+str(model_output.get_shape())+'\n')

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        print_info('Loading weights...')
        saver.restore(sess, weights_path)
        print_ok('Done!\n')

        all_inps = glob.glob(inp_path+'*.jpg')
        if len(all_inps) == 0:
            print_err('ERR: No jpg images found in '+inp_path+'\n')
            quit()
        print_info('Found %d images!'%len(all_inps))

        inv_index = ()
        all_descriptors = np.zeros((0,phoc_size))

        # process each image independently
        for i, inp in enumerate(all_inps):

            progress = 100.*(i+1)/len(all_inps)
            print_progress(progress, tcolors.INFO+'Processing images (%d/%d)'%(i+1,len(all_inps))+tcolors.ENDC)

            inp_feed = np.expand_dims(img_preprocess(inp, shape=img_shape, letterbox=True), 0)
            feed_dict = {model_input : inp_feed}
            out = sess.run(model_output, feed_dict)

            # activate the outputs
            descriptors = out.reshape((-1,phoc_size+5))
            descriptors = expit(descriptors)
            # filter low confidence descriptors
            valid_descriptors = descriptors[tuple(np.where(descriptors[:,4] > thresh)[0]), 5:]

            # add to NN inverted index with the image filename as key
            num_valid_descriptors = valid_descriptors.shape[0]
            inv_index += (i,)*num_valid_descriptors
            all_descriptors.resize((all_descriptors.shape[0]+num_valid_descriptors, phoc_size))
            all_descriptors[all_descriptors.shape[0]-num_valid_descriptors:,:] = valid_descriptors

        # build NN index
        print_info('\n\nBuilding NN (sklearn ball_tree) index with %d descriptors... '%all_descriptors.shape[0])
        sys.stdout.flush()
        nbrs = NearestNeighbors(algorithm='ball_tree', metric='euclidean').fit(all_descriptors)
        print_ok('Done!\n\n')

        # prepare queries (uses the GT format of IIIT_STR and Sports10K datasets)
        gt_data = io.matlab.loadmat(gt_data_path)
        str_queries = []
        for i in range(gt_data['data'].shape[1]):
            str_queries.append(str(gt_data['data'][0,i][0][0][0][0]).lower())
        queries = np.zeros((len(str_queries),phoc_size))
        for i,query in enumerate(str_queries):
            queries[i,:] = np.array(build_phoc(query)).reshape(1,-1)

        # do NN search
        print_info('Do NN search for %d query descriptors... '%len(str_queries))
        sys.stdout.flush()
        distances, indices = nbrs.kneighbors(queries, n_neighbors=n_neighbors)
        print_ok('Done!\n\n')

        # Calculate AP for each query (uses the GT format of IIIT_STR and Sports10K datasets)
        print_info('Calculating mAP...\n')
        mAP = 0.
        top10_paths = {}
        for i in range(len(str_queries)):
            query = str_queries[i]
 
            rel = []
            for j in range(len(gt_data['data'][0,i][1])):
                rel.append(str(gt_data['data'][0,i][1][j][0][0]).lower())
            res_images = [all_inps[inv_index[indices[i,j]]] for j in range(indices.shape[1])]
            res_scores = 1 - (distances[i,:] / np.max(distances[i,:]))
 
            y_true   = np.zeros((len(all_inps),))
            y_scores = np.zeros((len(all_inps),))
            for r in rel:
                y_true[all_inps.index(inp_path+r)] = 1
            for n,r in enumerate(res_images):
                y_scores[all_inps.index(r)] = np.max((y_scores[all_inps.index(r)], res_scores[n]))
        
            ap = average_precision_score(y_true, y_scores)
            print('  Average precision (AP) for query "'+str_queries[i]+'"= '+str(ap))
            mAP += ap

            # keep the paths of top10 images for later visualization
            top10_paths[str_queries[i]] = [all_inps[j] for j in y_scores.argsort()[::-1][:10]]

        mAP /= len(str_queries) 
        print_ok('Final mean Average Precision (mAP) = '+str(mAP)+'\n')


        # save top10 results as json
        import json
        with open('top10_results.json','w') as fp:
            json.dump(top10_paths, fp)
        print_info('Saved Top-10 results for each query as top10_results.json\n')
