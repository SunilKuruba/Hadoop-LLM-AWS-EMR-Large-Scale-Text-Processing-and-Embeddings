import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingType
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.*
import org.apache.hadoop.mapred.*
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory

import java.io.{File, IOException}
import java.util
import scala.jdk.CollectionConverters.*
import java.nio.file.{Paths, Files}
import java.nio.charset.StandardCharsets

object Tokenizer {

  class TokenizerMapper extends MapReduceBase with Mapper[LongWritable, Text, Text, IntWritable] {
    private final val one = new IntWritable(1)
    private val word = new Text()
    private val encoding = Encodings.newDefaultEncodingRegistry().getEncoding(EncodingType.CL100K_BASE)

    @throws[IOException]
    override def map(key: LongWritable, value: Text, output: OutputCollector[Text, IntWritable], reporter: Reporter): Unit =
      val tokens = encoding.encode(value.toString)
      tokens.asScala.foreach { token =>
        word.set(token.toString)
        output.collect(word, one)
      }
  }

  class IntSumReducer extends MapReduceBase with Reducer[Text, IntWritable, Text, IntWritable] {
    override def reduce(key: Text, values: util.Iterator[IntWritable], output: OutputCollector[Text, IntWritable], reporter: Reporter): Unit =
      val sum = values.asScala.map(_.get()).sum;
      output.collect(key, new IntWritable(sum))
  }

  //  @main
  def tokenizationMain(inputPath: String, outputPath: String): RunningJob = {
    val conf: JobConf = new JobConf(this.getClass)
    conf.setJobName("WordCount")
    conf.set("fs.defaultFS", "hdfs://localhost:9000")
    //    conf.set("mapreduce.job.maps", "5")
    conf.set("mapreduce.job.reduces", "5")
    // Set the maximum split size
    conf.setLong("mapreduce.input.fileinputformat.split.maxsize", 6710) // 64 MB
    conf.setOutputKeyClass(classOf[Text])
    conf.setOutputValueClass(classOf[IntWritable])
    conf.setMapperClass(classOf[TokenizerMapper])
    conf.setCombinerClass(classOf[IntSumReducer])
    conf.setReducerClass(classOf[IntSumReducer])
    conf.setInputFormat(classOf[TextInputFormat])
    conf.setOutputFormat(classOf[TextOutputFormat[Text, IntWritable]])
    FileInputFormat.setInputPaths(conf, new Path(inputPath))
    FileOutputFormat.setOutputPath(conf, new Path(outputPath))
    JobClient.runJob(conf)
  }

  @main def test(): Unit = {
    val tokenizer = Encodings.newDefaultEncodingRegistry().getEncoding(EncodingType.CL100K_BASE)

    val input = "Hello world!"
    val tokenIds = tokenizer.encode(input)
    val output =  tokenizer.decode(tokenIds)

    println(s"Original Text: $input")
    println(s"Token IDs: $tokenIds")
    println(s"Original text back: $output")

    val file2 = "src/main/resources/test_input_2.txt";
    Files.write(Paths.get(file2), tokenIds.toString.getBytes(StandardCharsets.UTF_8))

    val file = new File("src/main/resources/test_input.txt") // Replace with your file path
    val sentenceIterator = new LineSentenceIterator(file)

    // Tokenizer configuration
    val tokenizerFactory = new DefaultTokenizerFactory()

    // Build Word2Vec model
    val word2Vec = new Word2Vec.Builder()
      .minWordFrequency(1) // Minimum frequency of words to be included
      .iterations(10) // Number of training iterations
      .layerSize(10) // Size of the word vectors
      .seed(42)
      .windowSize(5) // Context window size for embeddings
      .iterate(sentenceIterator)
      .tokenizerFactory(tokenizerFactory)
      .build()

    // Train the model
    word2Vec.fit()

    // Save the model for later use
//    WordVectorSerializer.writeWord2VecModel(word2Vec, new File("word2vec_model.bin"))

    // Get embedding for a token
    val embedding: Array[Double] = word2Vec.getWordVector("Hello") // Replace "example" with your token

    // Print embedding
    if (embedding != null) {
      println("Embedding for 'example': ")
      embedding.foreach(value => print(value + " "))
    } else {
      println("Word not in the vocabulary!")
    }
  }
}
