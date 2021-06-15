import tensorflow as tf
# seq = 1
# features_seq = tf.train.Feature(int64_list=tf.train.Int64List(value=[seq]))
# feature_lists = tf.train.FeatureLists(feature_list={'features_seq':tf.train.FeatureList(feature=[features_seq])})
# example = tf.train.SequenceExample(feature_lists=feature_lists)
# seq_writer = tf.io.TFRecordWriter("seq.tfrecord")
# seq_writer.write(example.SerializeToString())
# raw_dataset = tf.data.TFRecordDataset("seq.tfrecord")
#
# for raw_record in raw_dataset.take(1):
#     example = tf.train.SequenceExample()
#     example.ParseFromString(raw_record.numpy())
#     print(example)




# seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
# features_seq = tf.train.Feature(int64_list=tf.train.Int64List(value=seq))
# feature_lists = tf.train.FeatureLists(feature_list={'features_seq':tf.train.FeatureList(feature=[features_seq])})
# example = tf.train.SequenceExample(feature_lists=feature_lists)
# seq_writer = tf.io.TFRecordWriter("seq_1.tfrecord")
# seq_writer.write(example.SerializeToString())
# raw_dataset = tf.data.TFRecordDataset("seq_1.tfrecord")
#
# for raw_record in raw_dataset.take(1):
#     example = tf.train.SequenceExample()
#     example.ParseFromString(raw_record.numpy())
#     print(example)

# seq_list = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [1, 2, 3]]
# seq_writer = tf.io.TFRecordWriter('seq_2.tfrecord')
# for seq in seq_list:
#     features_seq = tf.train.Feature(int64_list=tf.train.Int64List(value=seq))
#     feature_lists = tf.train.FeatureLists(feature_list={'features_seq': tf.train.FeatureList(feature=[features_seq])})
#     example = tf.train.SequenceExample(feature_lists=feature_lists)
#     seq_writer.write(example.SerializeToString())
#
# raw_dataset = tf.data.TFRecordDataset("seq_1.tfrecord")
# for raw_record in raw_dataset.take(1):
#     example = tf.train.SequenceExample()
#     example.ParseFromString(raw_record.numpy())
#     print(example)


seq_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
seq_2 = [0, 9, 8, 7, 6, 5, 4, 3, 2, 1]
features_seq_1  = tf.train.Feature(int64_list=tf.train.Int64List(value=seq_1))
features_seq_2 = tf.train.Feature(int64_list=tf.train.Int64List(value=seq_2))
feature_lists = tf.train.FeatureLists(feature_list={
    'features_seq_1': tf.train.FeatureList(feature=[features_seq_1]),
    'features_seq_2': tf.train.FeatureList(feature=[features_seq_2])})
example = tf.train.SequenceExample(feature_lists=feature_lists)
seq_writer = tf.io.TFRecordWriter('seq_3.tfrecord')
seq_writer.write(example.SerializeToString())
raw_dataset = tf.data.TFRecordDataset("seq_3.tfrecord")
for raw_record in raw_dataset.take(1):
    example = tf.train.SequenceExample()
    example.ParseFromString(raw_record.numpy())
    print(example)
