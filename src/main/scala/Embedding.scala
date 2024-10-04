import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.{Encoding, EncodingRegistry, EncodingType}
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
import org.nd4j.linalg.ops.transforms.Transforms

import java.io.IOException
import scala.io.Source
import scala.jdk.CollectionConverters.*

// TODO: implement config file
// TODO: add tests
// TODO: add java doc
// remove var
// remove foreach, refactor
// update ReadME
object Embedding {

  class EmbeddingMapper extends MapReduceBase with Mapper[LongWritable, Text, Text, Text] {
    private val outputKey = new Text()

    @throws[IOException]
    override def map(key: LongWritable, value: Text, output: OutputCollector[Text, Text], reporter: Reporter): Unit =
      val sentences = value.toString.split("\n").toList
      val tokenizedSentences: List[List[Integer]] = sentences.map(sentence => Tokenizer.encode(sentence).asScala.toList)

      // Step 3: Prepare input and labels for "next word prediction"
      val vocabSize = 100_000
      val embeddingDim = 5 // Set embedding dimensions

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
        .nIn(vocabSize + 1) // +1 to include padding token if necessary
        .nOut(embeddingDim) // Embedding dimension
        .activation(Activation.IDENTITY) // No activation function
        .build())
       .layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT) // Sparse cross-entropy for classification
        .nIn(embeddingDim)
        .nOut(vocabSize + 1) // Output is a probability distribution over the vocabulary
        .activation(Activation.SOFTMAX) // Softmax for next word prediction
        .build())
        .build()

      val model = new MultiLayerNetwork(config)
      model.init()

      // Step 5: Train the model
      val numEpochs = 1 // Number of training epochs
      for (_ <- 0 until numEpochs) {
       model.fit(inputFeatures, outputLabels)
      }

      // Step 6: Extract the learned embeddings for each token
      val embeddings: INDArray = model.getLayer(0).getParam("W")

      // Step 2: Get the number of rows (i.e., the vocabulary size) in the embedding matrix
      val numWords = embeddings.rows()

      flattenedTokens.foreach(t=> {
        val word = Tokenizer.decode(t)
        outputKey.set(word + "\t" + t)
        output.collect(outputKey, new Text(embeddings.getRow(t.longValue()).toStringFull))
      })

  }

  class EmbeddingReducer extends MapReduceBase with Reducer[Text, Text, Text, Text] {
    override def reduce(key: Text, values: java.util.Iterator[Text], output: OutputCollector[Text, Text], reporter: Reporter): Unit = {
      val average = calculateAverage(values)
      output.collect(key, new Text(average.mkString("[", ", ", "]")))
    }
  }

  def parseArray(text: Text): Array[Float] = {
    // Convert Text to String and parse it to Array[Float]
    text.toString
      .replace("[", "") // Remove the opening bracket
      .replace("]", "") // Remove the closing bracket
      .split(",") // Split by comma
      .map(_.trim.toFloat) // Trim whitespace and convert to Float
  }

  def calculateAverage(iterator: java.util.Iterator[Text]): Array[Float] = {
    // Initialize variables to store sum and count of arrays
    var sumArray: Array[Float] = Array.empty
    var count = 0

    // Iterate over all Text entries
    while (iterator.hasNext) {
      val currentArray = parseArray(iterator.next())

      // Initialize sumArray if it's the first time
      if (sumArray.isEmpty) {
        sumArray = new Array[Float](currentArray.length)
      }

      // Add currentArray to sumArray element-wise
      for (i <- currentArray.indices) {
        sumArray(i) += currentArray(i)
      }

      count += 1
    }

    // Calculate the average for each element
    for (i <- sumArray.indices) {
      sumArray(i) /= count
    }

    sumArray
  }

  @main
  def embeddingMain(inputPath: String, outputPath: String): RunningJob = {
    val conf: JobConf = new JobConf(this.getClass)
    conf.setJobName("Embedder")
    conf.set("fs.defaultFS", "hdfs://localhost:9000")
    // Set the maximum split size
    //    conf.setLong("mapreduce.input.fileinputformat.split.maxsize", 6710) // 64 MB
    conf.setOutputKeyClass(classOf[Text])
    conf.setOutputValueClass(classOf[Text])
    conf.setMapperClass(classOf[EmbeddingMapper])
    conf.setReducerClass(classOf[EmbeddingReducer])
    conf.setInputFormat(classOf[TextInputFormat])
    conf.setOutputFormat(classOf[TextOutputFormat[Text, Text]])
    FileInputFormat.setInputPaths(conf, new Path(inputPath))
    FileOutputFormat.setOutputPath(conf, new Path(outputPath))
    JobClient.runJob(conf)
  }

  @main
  def embeddingTest(): Unit = {

    // Step 1: Read text file
    val inputFilePath = "src/main/resources/test_input.txt"
    val sentences = Source.fromFile(inputFilePath).getLines().toList

    // Step 2: Tokenization using JTokkit
    val encodingRegistry: EncodingRegistry = Encodings.newDefaultEncodingRegistry()
    val encoding: Encoding = encodingRegistry.getEncoding(EncodingType.CL100K_BASE) // Using CL100K_BASE encoding

    // Tokenize the sentences and convert them into token IDs (integers)
    val tokenizedSentences: List[List[Integer]] = sentences.map(sentence => encoding.encode(sentence).asScala.toList)

    // Step 3: Prepare input and labels for "next word prediction"
    val vocabSize = 100_000
    val embeddingDim = 5 // Set embedding dimensions

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
        .nIn(vocabSize + 1) // +1 to include padding token if necessary
        .nOut(embeddingDim) // Embedding dimension
        .activation(Activation.IDENTITY) // No activation function
        .build())
      .layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT) // Sparse cross-entropy for classification
        .nIn(embeddingDim)
        .nOut(vocabSize + 1) // Output is a probability distribution over the vocabulary
        .activation(Activation.SOFTMAX) // Softmax for next word prediction
        .build())
      .build()

    val model = new MultiLayerNetwork(config)
    model.init()

    // Step 5: Train the model
    val numEpochs = 100 // Number of training epochs
    for (_ <- 0 until numEpochs) {
      model.fit(inputFeatures, outputLabels)
    }

    // Step 6: Extract the learned embeddings for each token
    val embeddings: INDArray = model.getLayer(0).getParam("W")

    // Step 7: Print out the learned embeddings
    println("Learned Embeddings:\n" + embeddings)

    // Step 2: Get the number of rows (i.e., the vocabulary size) in the embedding matrix
    val numWords = embeddings.rows()

    // Step 3: Iterate through each row (i.e., each word's embedding)
    for (i <- 0 until numWords) {
      val embeddingVector = embeddings.getRow(i) // Get the embedding vector for word i
      val word = Tokenizer.decode(i)
      val output = ArrayWritable(classOf[FloatWritable], embeddingVector.toFloatVector.map(value => new FloatWritable(value)))

      println(s"Word: $word Token ID: $i, Embedding: ${embeddingVector}")
    }

    // Step 1: Get the token ID for "king" and "queen"
    val kingTokenId = encoding.encode("Hello").asScala.head // Token ID for "king"
    val queenTokenId = encoding.encode("world").asScala.head // Token ID for "queen"

    // Step 2: Get the embeddings for "king" and "queen"
    val sampleEmbedding1: INDArray = embeddings.getRow(kingTokenId.longValue())
    val sampleEmbedding2: INDArray = embeddings.getRow(queenTokenId.longValue())

    // Step 3: Compute cosine similarity
    val similarity = Transforms.allCosineSimilarities(sampleEmbedding1, sampleEmbedding2)

    // Step 4: Print the result
    println(s"Cosine similarity between 'Hello' and 'world': $similarity")
  }
}
