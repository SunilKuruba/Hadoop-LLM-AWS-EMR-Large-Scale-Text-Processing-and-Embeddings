import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.*;
import org.scalatest.flatspec.AnyFlatSpec;

class TokenizerSpec extends AnyFlatSpec with Matchers {

  "TokenizerMapper" should "output token and encoded value" in {
    val mapper = new Tokenizer.TokenizerMapper()
    val mockOutputCollector = mock(classOf[OutputCollector[Text, IntWritable]])
    val mockReporter = mock(classOf[Reporter])

    val inputKey = new LongWritable(1)
    val inputValue = new Text("Hello World")

    mapper.map(inputKey, inputValue, mockOutputCollector, mockReporter)

    val captor = ArgumentCaptor.forClass(classOf[Text])
    verify(mockOutputCollector, times(2)).collect(captor.capture(), any(classOf[IntWritable]))

    val outputKeys = captor.getAllValues
    outputKeys.get(0).toString should include("hello")
    outputKeys.get(1).toString should include("world")
  }

  "IntSumReducer" should "sum the values for a given key" in {
    val reducer = new Tokenizer.IntSumReducer()
    val mockOutputCollector = mock(classOf[OutputCollector[Text, IntWritable]])
    val mockReporter = mock(classOf[Reporter])

    val inputKey = new Text("test")
    val inputValues = List(new IntWritable(1), new IntWritable(2), new IntWritable(3)).asJava

    reducer.reduce(inputKey, inputValues.iterator(), mockOutputCollector, mockReporter)

    verify(mockOutputCollector).collect(inputKey, new IntWritable(6))
  }

  "Tokenizer job" should "complete successfully" in {
    val inputPath = "test_input.txt"
    val outputPath = "test_output"

    // Create input file
    Files.write(Paths.get(inputPath), "Hello Hadoop\nHadoop is fun".getBytes)

    // Set up job configuration
    val conf = new JobConf(classOf[Tokenizer])
    FileInputFormat.setInputPaths(conf, new Path(inputPath))
    FileOutputFormat.setOutputPath(conf, new Path(outputPath))

    // Run job
    val job = Tokenizer.tokenizerMain(inputPath, outputPath)

    job.isComplete shouldBe true
    job.isSuccessful shouldBe true

    // Check if the output exists and contains expected results
    val outputDir = new Path(outputPath)
    val fs = outputDir.getFileSystem(conf)
    fs.exists(outputDir) shouldBe true
  }
}