class TrackSentiment:
    """
    Class which track sentiments. You pass it a list of keys for which to track
    the minimum, maximum, and sum and then if you pass it a sentiment dictionary
    it will track the minimums, maximums, and sums for the keys whose names match
    the values in the lists you initialized it with. The object also keeps track
    of the total number of sentiments it has processed so that you can get an
    average later.
    """

    def __init__(self, sum_keys, max_keys, min_keys, include_zeros_in_averages):
        self.total_sentiments = 0
        self.sums = {}
        self.mins = {}
        self.maxes = {}
        self.include_zeros_in_averages = include_zeros_in_averages
        for k in sum_keys:
            self.sums[k] = []
        for k in min_keys:
            self.mins[k] = 100
        for k in max_keys:
            self.maxes[k] = -100

    def process_sentiment(self, sentiment_dict):
        self.total_sentiments = self.total_sentiments + 1
        for k in self.sums.keys():
            if k in sentiment_dict:
                if sentiment_dict[k] == 0 and self.include_zeros_in_averages:
                    # if it is a zero we only include it if include_zeros_in_averages is true
                    self.sums[k].append(sentiment_dict[k])
                elif sentiment_dict[k] == 0 and not self.include_zeros_in_averages:
                    # the sentiment value is zero, but we are not including zeros, do nothing.
                    pass
                else:
                    self.sums[k].append(sentiment_dict[k])
        for k in self.mins.keys():
            if k in sentiment_dict:
                if self.mins[k] > sentiment_dict[k]:
                    self.mins[k] = sentiment_dict[k]
        for k in self.maxes.keys():
            if k in sentiment_dict:
                if self.maxes[k] < sentiment_dict[k]:
                    self.maxes[k] = sentiment_dict[k]

    def get_sentiment_summary(self):
        """
        Returns the sentiment summary. This is a dictionary that contains both the labels and the values
        for all sentiments that have been processed.
        """
        if self.total_sentiments == 0:
            raise RuntimeError(
                "No sentiments were processed for this object, the summary is invalid and would contain extreme "
                "starter values.")

        summary = {}
        for k, v in self.sums.items():
            new_key = k + "_avg"
            if len(v) > 0:
                new_value = sum(v) / len(v)
            else:
                new_value = 0
            summary[new_key] = new_value
        for k, v in self.mins.items():
            new_key = k + "_min"
            summary[new_key] = v
        for k, v in self.maxes.items():
            new_key = k + "_max"
            summary[new_key] = v

        return summary

    def has_sentiment_vector(self):
        """
        Returns True if this object contains sentiments, false otherwise.
        """
        if self.total_sentiments > 0:
            return True
        else:
            return False

    def get_sentiment_vector(self):
        """
        Returns the sentiment analysis as a list of values for all sentiments that have been processed.
        """
        if not self.has_sentiment_vector():
            raise ValueError(
                "No sentiments were processed for this object, the summary is invalid and would contain extreme "
                "starter values.")
        vector = []
        for k, v in self.sums.items():
            if len(v) > 0:
                new_value = sum(v) / len(v)
            else:
                new_value = 0
            vector.append(new_value)
        for k, v in self.mins.items():
            vector.append(v)
        for k, v in self.maxes.items():
            vector.append(v)
        return vector

    def get_sentiment_vector_labels(self):
        """
        Returns a list of label corresponding to the values in the sentiment vector.
        """
        labels = []
        for k, v in self.sums.items():
            labels.append(k + "_avg")
        for k, v in self.mins.items():
            labels.append(k + "_min")
        for k, v in self.maxes.items():
            labels.append(k + "_max")
        return labels
