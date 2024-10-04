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

import java.io.{DataInput, DataOutput, IOException}
import scala.io.Source
import scala.jdk.CollectionConverters.*


object Embedding {

  class EmbeddingMapper extends MapReduceBase with Mapper[LongWritable, Text, Text, ArrayWritable] {
    private val outputKey = new Text()

    @throws[IOException]
    override def map(key: LongWritable, value: Text, output: OutputCollector[Text, ArrayWritable], reporter: Reporter): Unit =
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

      // Step 3: Iterate through each row (i.e., each word's embedding)
      for (i <- 0 until numWords) {
        val embeddingVector = embeddings.getRow(i) // Get the embedding vector for word i
        val word = Tokenizer.decode(i)
        outputKey.set(word+"\t"+i)
        val arrayValue = ArrayWritable(classOf[FloatWritable], embeddingVector.toFloatVector.map(value => new FloatWritable(value)))
        output.collect(outputKey, arrayValue)
      }
  }

  class EmbeddingReducer extends MapReduceBase with Reducer[Text, ArrayWritable, Text, ArrayWritable] {
    override def reduce(key: Text, values: java.util.Iterator[ArrayWritable],
                        output: OutputCollector[Text, ArrayWritable], reporter: Reporter): Unit = {
      var sum: Array[Double] = null
      var count = 0

      // Iterate over the values to compute the sum of the arrays
      while (values.hasNext) {
        val arrayWritable = values.next()
        val array = arrayWritable.get.map(_.asInstanceOf[DoubleWritable].get)

        if (sum == null) {
          sum = new Array[Double](array.length)
        }

        for (i <- array.indices) {
          sum(i) += array(i)
        }
        count += 1
      }

      // Compute the average by dividing the sum by the count
      val avg = sum.map(_ / count)

      // Create an ArrayWritable with DoubleWritable values
      val resultArrayWritable = new ArrayWritable(classOf[DoubleWritable])
      resultArrayWritable.set(avg.map(new DoubleWritable(_)))

      // Write the key and average array to the output
      output.collect(key, resultArrayWritable)
    }
  }

  @main
  def test(): Unit ={


    val arrayValue = ArrayWritable(classOf[FloatWritable], Array(1,2,3,4,5).map(value => new FloatWritable(value)))
    val iterator = Iterator(arrayValue)
    iterator.foreach(x=> x)
//    System.out.println(arrayValue)
  }

  def convertIteratorToList(iterator: java.util.Iterator[ArrayWritable]): List[Array[Float]] = {
    // Convert java.util.Iterator to scala iterator
    import scala.jdk.CollectionConverters._
    val scalaIterator = iterator.asScala

    // Convert each ArrayWritable to Array[Float]
    scalaIterator.map { arrayWritable =>
      // Get the Array[Writable] from ArrayWritable and map to Array[Float]
      arrayWritable.get().map(_.asInstanceOf[FloatWritable].get().toFloat)
    }.toList
  }

  // Usage example
  def main(args: Array[String]): Unit = {
    // Sample input for testing
    val input: java.util.Iterator[ArrayWritable] = getSampleInput().asJava

    // Convert to List[Array[Float]]
    val resultList: List[Array[Float]] = convertIteratorToList(input)

    // Print the result
    resultList.foreach(arr => println(arr.mkString(", ")))
  }

  // Helper function to create sample input for testing
  def getSampleInput(): Iterator[ArrayWritable] = {
    val array1 = new ArrayWritable(classOf[FloatWritable], Array(new FloatWritable(1.0f), new FloatWritable(2.0f), new FloatWritable(3.0f)))
    val array2 = new ArrayWritable(classOf[FloatWritable], Array(new FloatWritable(4.0f), new FloatWritable(5.0f), new FloatWritable(6.0f)))
    val array3 = new ArrayWritable(classOf[FloatWritable], Array(new FloatWritable(7.0f), new FloatWritable(8.0f), new FloatWritable(9.0f)))

    Iterator(array1, array2, array3)
  }

  @main
  def embeddingMain(inputPath: String, outputPath: String): RunningJob = {
    val conf: JobConf = new JobConf(this.getClass)
    conf.setJobName("Embedder")
    conf.set("fs.defaultFS", "hdfs://localhost:9000")
    // Set the maximum split size
    //    conf.setLong("mapreduce.input.fileinputformat.split.maxsize", 6710) // 64 MB
    conf.setOutputKeyClass(classOf[Text])
    conf.setOutputValueClass(classOf[ArrayWritable])
    conf.setMapperClass(classOf[EmbeddingMapper])
    conf.setReducerClass(classOf[EmbeddingReducer])
    conf.setInputFormat(classOf[TextInputFormat])
    conf.setOutputFormat(classOf[TextOutputFormat[Text, ArrayWritable]])
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
