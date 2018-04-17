package org.bigDataCourse.recommenderSystem;

import java.io.File;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;

public class RecommenderEvaluatorIR {
	
	public static void main(String[] args) throws Exception
	{
	DataModel model = new FileDataModel(new File(args[0]));
	RecommenderIRStatsEvaluator evaluator = new GenericRecommenderIRStatsEvaluator();
	RecommenderBuilder builder = new MyRecommenderBuilder();
	IRStatistics stats = evaluator.evaluate(builder,null, model,null, 10,  
			GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0);
	System.out.println("precision: "+stats.getPrecision());
	System.out.println("Recall: "+stats.getRecall());
	System.out.println("F1 measure: "+ stats.getF1Measure());
	
}


}
