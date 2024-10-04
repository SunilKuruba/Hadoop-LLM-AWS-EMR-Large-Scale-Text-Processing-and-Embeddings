import com.typesafe.config.ConfigFactory
import org.apache.hadoop.fs.Path
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
import scala.jdk.CollectionConverters.*

// TODO: add tests
// TODO: add java doc
// TODO: update README
object Embedding {
  private val logger: Logger = LoggerFactory.getLogger(Tokenizer.getClass)
  private val outputKey = new Text()
  private val outputValue = new Text()
  private val appConfig = ConfigFactory.load

  class EmbeddingMapper extends MapReduceBase with Mapper[LongWritable, Text, Text, Text] {
    @throws[IOException]
    override def map(key: LongWritable, value: Text, output: OutputCollector[Text, Text], reporter: Reporter): Unit =
      logger.info(s"Started running Embedding Mapper with key: $key")
      
      val sentences = value.toString.split("\n").toList
      val tokenizedSentences: List[List[Integer]] = sentences.map(sentence => Tokenizer.encode(sentence).asScala.toList)

      // Prepare input and labels by flattening sentences into individual tokens
      val flattenedTokens = tokenizedSentences.flatMap(tokens => tokens.dropRight(1)) // Input: [w1, w2, ..., wn-1]
      val flattenedLabels = tokenizedSentences.flatMap(tokens => tokens.drop(1)) // Label: [w2, w3, ..., wn]

      // Convert flattened lists to INDArray format
      val inputFeatures: INDArray = Nd4j.create(flattenedTokens.map(_.toFloat).toArray, Array(flattenedTokens.size, 1))
      val outputLabels: INDArray = Nd4j.create(flattenedLabels.map(_.toFloat).toArray, Array(flattenedLabels.size, 1))

      // A Neural Network with Embedding layer and Output Layer (softmax for prediction)
       val config: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
       .list()
       .layer(new EmbeddingLayer.Builder()
        .nIn(appConfig.getInt("embeddingJob.vocabSize") + 1) // +1 to include padding token if necessary
        .nOut(appConfig.getInt("embeddingJob.embeddingDim")) // Embedding dimension
        .activation(Activation.IDENTITY) // No activation function
        .build())
       .layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT) // Sparse cross-entropy for classification
        .nIn(appConfig.getInt("embeddingJob.embeddingDim"))
        .nOut(appConfig.getInt("embeddingJob.vocabSize") + 1) // Output is a probability distribution over the vocabulary
        .activation(Activation.SOFTMAX) // Softmax for next word prediction
        .build())
        .build()

      val model = new MultiLayerNetwork(config)
      model.init()

      // Train the model
      logger.info(s"Model training started for key: $key")
      val numEpochs = appConfig.getInt("embeddingJob.testNumEpochs")
      (0 until numEpochs).foreach { i =>
        if(i == numEpochs/2) logger.info(s"Model training 50% completed for key: $key")
        model.fit(inputFeatures, outputLabels)
      }

      // Extract the learned embeddings for each token
      val embeddings: INDArray = model.getLayer(0).getParam("W")

      flattenedTokens.foreach(token=> {
        val word = Tokenizer.decode(token)
        outputKey.set(word + "\t" + token)
        outputValue.set(embeddings.getRow(token.longValue()).toStringFull)
        output.collect(outputKey, outputValue)
      })
  }

  class EmbeddingReducer extends MapReduceBase with Reducer[Text, Text, Text, Text] {
    override def reduce(key: Text, values: java.util.Iterator[Text], output: OutputCollector[Text, Text], reporter: Reporter): Unit = {
      val average = Utility.calculateAverage(values)
      outputValue.set(average.mkString("[", ", ", "]"))
      output.collect(key, outputValue)
    }
  }

  @main
  def embeddingMain(inputPath: String, outputPath: String): RunningJob = {
    val config = ConfigFactory.load
    val jobConf: JobConf = new JobConf(this.getClass)
    jobConf.setJobName(config.getString("embeddingJob.jobName"))
    jobConf.set("fs.defaultFS", config.getString("hadoop.fs.defaultFS"))
    jobConf.setLong("mapreduce.input.fileinputformat.split.maxsize", config.getLong("hadoop.blockSize"))
    jobConf.setOutputKeyClass(classOf[Text])
    jobConf.setOutputValueClass(classOf[Text])
    jobConf.setMapperClass(classOf[EmbeddingMapper])
    jobConf.setReducerClass(classOf[EmbeddingReducer])
    jobConf.setInputFormat(classOf[TextInputFormat])
    jobConf.setOutputFormat(classOf[TextOutputFormat[Text, Text]])
    FileInputFormat.setInputPaths(jobConf, new Path(inputPath))
    FileOutputFormat.setOutputPath(jobConf, new Path(outputPath))

    logger.info("Starting the MapReduce job")
    val job = JobClient.runJob(jobConf)
    logger.info("Job completed successfully")
    job
  }
}
