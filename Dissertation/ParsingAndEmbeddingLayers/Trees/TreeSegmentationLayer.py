import tensorflow as tf

class TreeSegmentationLayer:
    def __init__(self):
        """
        The Tree Segmentation layer class where the functions for carrying out 
        segmentation on trees are declared
        """
        pass

    def segmentationFunction(self, segmentationFunction: str):
        """
        Determine the type of segmentation function to use

        segmentationFunction: str - The string representation of the chosen segmentation function

        Returns
        The tensorflow function corresponding to 'segmentationFunction'
        """
        segmentationFunction = segmentationFunction.split("_")
        if segmentationFunction[0] == "sorted":
            if segmentationFunction[1] == "sum":
                return tf.math.segment_sum
            if segmentationFunction[1] == "mean":
                return tf.math.segment_mean
            if segmentationFunction[1] == "max":
                return tf.math.segment_max
            if segmentationFunction[1] == "min":
                return tf.math.segment_min
            if segmentationFunction[1] == "prod":
                return tf.math.segment_prod
        elif segmentationFunction[0] == "unsorted":
            if segmentationFunction[1] == "sum":
                return tf.math.unsorted_segment_sum
            if segmentationFunction[1] == "mean":
                return tf.math.unsorted_segment_mean
            if segmentationFunction[1] == "max":
                return tf.math.unsorted_segment_max
            if segmentationFunction[1] == "min":
                return tf.math.unsorted_segment_min
            if segmentationFunction[1] == "prod":
                return tf.math.unsorted_segment_prod
        else:
            return None

    def segmentationLayer(self, segmentationFunction: str, nodeEmbeddings: tf.Tensor, numSegments: int):
        """
        The segmentation function proper where the tree is segmented

        segmentationFunction: str - The string representation of the segmentation function
        nodeEmbeddings: tf.Tensor - The nodes to be segmented
        numSegments: int - The number of segments to be used

        Returns
        The segmented representation of 'nodeEmbeddings'
        """
        seg = segmentationFunction.lower()
        segmentationFunction = self.segmentationFunction(segmentationFunction)

        if seg.split("_")[0] == "unsorted":
            return segmentationFunction(nodeEmbeddings, tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 
            32, 33, 34, 35, 36, 37, 38, 39]), numSegments)
        else:
            return segmentationFunction(nodeEmbeddings, tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 
            32, 33, 34, 35, 36, 37, 38, 39]))
