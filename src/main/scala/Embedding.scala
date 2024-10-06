import com.typesafe.config.ConfigFactory
import org.apache.hadoop.io.*
import org.apache.hadoop.mapred.*
import org.deeplearning4j.nn.conf.layers.{EmbeddingLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}

import java.io.IOException
import java.time.Instant
import scala.jdk.CollectionConverters.*

/**
 * The Embedding object processes input text using Hadoop's MapReduce framework
 * to generate word embeddings using a neural network with an embedding layer.
 * It encodes sentences into tokens and trains a model to learn embeddings.
 */
object Embedding {

  /** Logger for logging important events and errors during the embedding process. */
  private val logger: Logger = LoggerFactory.getLogger(Tokenizer.getClass)

  /** Key and value used for output during MapReduce tasks. */
  private val outputKey = new Text()
  private val outputValue = new Text()

  /** Configuration object to load application settings. */
  private val appConfig = ConfigFactory.load

  /**
   * EmbeddingMapper is a MapReduce Mapper class. It tokenizes input text,
   * builds and trains a neural network to learn word embeddings, and emits the token
   * and its corresponding embedding.
   */
  class EmbeddingMapper extends MapReduceBase with Mapper[LongWritable, Text, Text, Text] {

    /**
     * Map function to process input text, tokenize it, train an embedding model, and
     * output the token along with its learned embedding.
     *
     * @param key      The key representing the line number of the input split.
     * @param value    The content of the input line.
     * @param output   The output collector to store the (token, embedding) pairs.
     * @param reporter Reporter for handling progress and status updates.
     * @throws IOException If any I/O error occurs during processing.
     */
    @throws[IOException]
    override def map(key: LongWritable, value: Text, output: OutputCollector[Text, Text], reporter: Reporter): Unit = {
      logger.info(s"Started running Embedding Mapper with key: $key at ${Instant.now()}")
      
      // Tokenize the input sentences
      val sentences = value.toString.trim.split("\n").toList.filter(_.nonEmpty)
      if(sentences.isEmpty) return
      val tokenizedSentences: List[List[Integer]] = sentences.map(sentence => Tokenizer.encode(sentence).asScala.toList)

      // Prepare input and label sequences
      val flattenedTokens = tokenizedSentences.flatMap(tokens => tokens.dropRight(1)) // Input: [w1, w2, ..., wn-1]
      val flattenedLabels = tokenizedSentences.flatMap(tokens => tokens.drop(1)) // Label: [w2, w3, ..., wn]

      // Convert flattened tokens to INDArray format for model training
      val inputFeatures: INDArray = Nd4j.create(flattenedTokens.map(_.toFloat).toArray, Array(flattenedTokens.size, 1))
      val outputLabels: INDArray = Nd4j.create(flattenedLabels.map(_.toFloat).toArray, Array(flattenedLabels.size, 1))

      // Build and configure the neural network
      val config: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
        .list()
        .layer(new EmbeddingLayer.Builder()
          .nIn(appConfig.getInt("embeddingJob.vocabSize") + 1) // Vocabulary size + 1 for padding
          .nOut(appConfig.getInt("embeddingJob.embeddingDim")) // Embedding dimension
          .activation(Activation.IDENTITY) // No activation function
          .build())
        .layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
          .nIn(appConfig.getInt("embeddingJob.embeddingDim"))
          .nOut(appConfig.getInt("embeddingJob.vocabSize") + 1) // Output: probability distribution over vocabulary
          .activation(Activation.SOFTMAX) // Softmax for classification
          .build())
        .build()

      val model = new MultiLayerNetwork(config)
      model.init()

      // Train the model
      logger.info(s"Model training started for key: $key")
      val numEpochs = appConfig.getInt("embeddingJob.numEpochs")
      (0 until numEpochs).foreach { i =>
        if(i == numEpochs / 2) logger.info(s"Model training 50% completed for key: $key")
        model.fit(inputFeatures, outputLabels)
      }

      // Extract embeddings from the trained model
      val embeddings: INDArray = model.getLayer(0).getParam("W")

      // Emit token and its learned embedding
      flattenedTokens.foreach(token => {
        val word = Tokenizer.decode(token)
        val outputKey = new Text(word + "\t" + token)
        val outputValue = new Text(embeddings.getRow(token.longValue()).toStringFull)
        output.collect(outputKey, outputValue)
      })
      logger.info(s"Ending Embedding Mapper with key: $key at ${Instant.now()}")
    }
  }

  /**
   * EmbeddingReducer is a MapReduce Reducer class. It aggregates the embeddings of each token,
   * computes the average embedding, and outputs the token with its averaged embedding.
   */
  class EmbeddingReducer extends MapReduceBase with Reducer[Text, Text, Text, Text] {

    /**
     * Reduce function to calculate the average embedding for each token and output the result.
     *
     * @param key      The token for which embeddings were generated.
     * @param values   An iterator over the embeddings for the token.
     * @param output   The output collector to store the (token, averaged embedding) pair.
     * @param reporter Reporter for handling progress and status updates.
     */
    override def reduce(key: Text, values: java.util.Iterator[Text], output: OutputCollector[Text, Text], reporter: Reporter): Unit = {
      val average = Utility.calculateAverage(values)
      outputValue.set(average.mkString("[", ", ", "]"))
      output.collect(key, outputValue)
    }
  }

  /**
   * The main entry point for the MapReduce job. It sets up the job configuration,
   * including the Mapper and Reducer classes, and starts the MapReduce job to learn embeddings.
   *
   * @return The RunningJob object representing the MapReduce job.
   */
  @main
  def embeddingMain(): RunningJob = {
    val jobConf: JobConf = JobConfig.createJob("embeddingJob.jobName")
    jobConf.setOutputKeyClass(classOf[Text])
    jobConf.setOutputValueClass(classOf[Text])
    jobConf.setMapperClass(classOf[EmbeddingMapper])
    jobConf.setReducerClass(classOf[EmbeddingReducer])
    jobConf.setInputFormat(classOf[TextInputFormat])
    jobConf.setOutputFormat(classOf[TextOutputFormat[Text, Text]])

    logger.info("Starting the MapReduce job")
    val job = JobClient.runJob(jobConf)
    logger.info("Job completed successfully")
    job
  }
}
