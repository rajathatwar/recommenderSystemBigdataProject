import org.apache.spark.mllib.recommendation._
import org.apache.spark.SparkContext._
import org.apache.spark.SparkContext
import org.apache.spark.rdd._


import org.apache.spark.SparkConf
object recommender {
  def main(args: Array[String]): Unit = {
    if (args.length < 2){
        System.err.println("Usage: symmetric app <input file>  <output file>")
        System.exit(1);  }
    
  /**
   * A function to compute recall at k for a single user  
   */
    def recallAtK(actual:Array[Int] , recommended:Array[Int], k:Int ):Double={
     val actual_set= actual.toSet
      val recommended_set= recommended.take(k).toSet
      return actual_set.intersect(recommended_set).size.toDouble/actual_set.size
      }
    /**
     * A function to compute precision at k for a single user
     */
    def precisionAtK(actual:Array[Int] , recommended:Array[Int], k:Int ):Double={
     val actual_set= actual.toSet
      val recommended_set= recommended.take(k).toSet
      return actual_set.intersect(recommended_set).size.toDouble/recommended_set.size
      }
    /**
     * A function to compute averag recall at k for all users
     */
    def averageRecallAtK(actualAndPredicted:RDD[(Array[Int],Array[Int])], k:Int):Double={
	      actualAndPredicted.map{case(actual,predicted)=>recallAtK(actual,predicted,k)}.mean}
	  /**
	   * A function to compute average precision at k for all users  
	   */
    def averagePrecisionAtK(actualAndPredicted:RDD[(Array[Int],Array[Int])], k:Int):Double={
	      actualAndPredicted.map{case(actual,predicted)=>precisionAtK(actual,predicted,k)}.mean}
    
    /**
     * A function to compute RMSE
     */
    def computeRMSE(model:MatrixFactorizationModel,testing:RDD[Rating]):Double={
     val userproducts=testing.map{case Rating(user,product,rate)=>(user,product)}
     val predictions=model.predict(userproducts).map{case Rating(user,product,rate)=>((user,product),rate)}
     val actual = testing.map{case Rating(user,product,rate)=>((user,product),rate)}
     val joined = actual.join(predictions)
     val mse= joined.map{case ((user,product),(actual,predicted))=>val err=actual-predicted; err*err}.mean
     scala.math.sqrt(mse)
     }
    
     //Creating a spark context
    val conf = new SparkConf().setAppName("symmetric app").setMaster("local[2]")
    val sc = new SparkContext(conf)
    
    //creating an rdd from the ratings file
    val ratingsrdd = sc.textFile(args(0)).map(line=>line.split("::")).map(splits=>Rating(splits(0).toInt,splits(1).toInt,splits(2).toDouble))
    
    //Splitting the data to training and testing sets
    val Array(training,testing)=ratingsrdd.randomSplit(Array(0.8,0.2))
    
    //Initializing ALs parameters
    val rank=40; val lambda=0.1; val numIterations=20
    val  model = ALS.train(training,rank,numIterations,lambda)

    //Computing RMSE
    println("RMSE is: "+computeRMSE(model, testing))
    

  //Computing average Precision and Recall at K
    val k=20; val threshold= 3.0
    //Find top 20 recommendations by the model
    val predictedRatings = model.recommendProductsForUsers(20)
    //filter the recommendations and keep only the ones with predicted rating >=threshold.Omit users with empty recommendations
    val predicted_filtered = predictedRatings.map{case(user,rec)=>(user, rec.filter(r=>r.rating>=threshold))}.filter{case(user,rec)=>rec.size>0}.map{case(user,rec)=>(user,rec.map(r=>r.product))}
    //filter the testing and keep only the ratings which are greater than threshold.Omit users with empty products.
    val testing_filtered= testing.filter(r=>r.rating>=threshold).map(r=>(r.user,r.product)).groupByKey.filter{case(user,products)=>products.size>0}
    //join the predicted and testing sets
    val joined =testing_filtered.join(predicted_filtered).map{case(user, (actual,predicted))=>(actual.toArray,predicted)}

    //compute the average precision and recall at K
    val avgPrecision= averagePrecisionAtK(joined,k)
    val avgRecall= averageRecallAtK(joined,k)
    println("average precision @ $k is: " +avgPrecision)
    println("average recall at $k is: "+ avgRecall)
    
    
    

}
}