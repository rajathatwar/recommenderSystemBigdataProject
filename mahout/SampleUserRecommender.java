package org.bigDataCourse.recommenderSystem;

import java.io.File;
import java.util.List;

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;


public class SampleUserRecommender {
	public static void main(String[] args) throws Exception
	{
		DataModel model = new FileDataModel(new File(args[0]));
		UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
		UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.9, similarity, model);
	UserBasedRecommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);
	List<RecommendedItem> recommendations = recommender.recommend(10, 10);
	for (RecommendedItem recommendation: recommendations)
		System.out.println(recommendation);
	
	
	
	}
}
