import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapred.{OutputCollector, Reporter}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class EmbeddingSpec extends AnyFlatSpec with Matchers {

  // Stub implementation of OutputCollector
  class TestOutputCollector extends OutputCollector[Text, Text] {
    var collectedData: List[(Text, Text)] = List.empty

    override def collect(key: Text, value: Text): Unit = {
      collectedData = collectedData :+ (key, value)
    }
  }

  // Stub implementation of Iterator
  class TestIterator(values: Seq[String]) extends java.util.Iterator[Text] {
    private var index = 0

    override def hasNext: Boolean = index < values.length

    override def next(): Text = {
      val nextValue = new Text(values(index))
      index += 1
      nextValue
    }
  }

  "EmbeddingMapper" should "correctly tokenize input and emit token embeddings" in {
    val mapper = new Embedding.EmbeddingMapper
    val outputCollector = new TestOutputCollector()
    val reporter = Reporter.NULL

    // Simulated input
    val inputKey = new LongWritable(1)
    val inputValue = new Text("This is a test sentence")

    // Call map function
    mapper.map(inputKey, inputValue, outputCollector, reporter)

    // Verify that outputCollector collected the expected token embeddings
    outputCollector.collectedData should not be empty
    outputCollector.collectedData.foreach { case (key, value) =>
      key.toString should include("\t") // Token and token id
      value.toString should include("[") // Embedding as an INDArray representation
    }
  }

  "EmbeddingReducer" should "correctly calculate the average embedding for a token" in {
    val reducer = new Embedding.EmbeddingReducer
    val outputCollector = new TestOutputCollector()
    val reporter =  Reporter.NULL

    // Input token and embeddings
    val inputKey = new Text("test")
    val embeddings = new TestIterator(Seq("[1.0, 2.0]", "[2.0, 3.0]"))

    // Call reduce function
    reducer.reduce(inputKey, embeddings, outputCollector, reporter)

    // Verify the output of the average embedding
    outputCollector.collectedData should have size 1
    val (key, value) = outputCollector.collectedData.head
    key.toString shouldBe "test"
    value.toString shouldBe "[1.5, 2.5]" // Average of [1.0, 2.0] and [2.0, 3.0]
  }

  "EmbeddingMapper" should "not emit any tokens or embeddings when input is empty" in {
    val mapper = new Embedding.EmbeddingMapper
    val outputCollector = new TestOutputCollector()
    val reporter = Reporter.NULL

    // Empty input
    val inputKey = new LongWritable(1)
    val inputValue = new Text("")

    // Call map function
    mapper.map(inputKey, inputValue, outputCollector, reporter)

    // Verify that no tokens were emitted
    outputCollector.collectedData shouldBe empty
  }

  "EmbeddingMapper" should "not emit any embeddings for a single word input" in {
    val mapper = new Embedding.EmbeddingMapper
    val outputCollector = new TestOutputCollector()
    val reporter = Reporter.NULL

    // Input with a single word
    val inputKey = new LongWritable(1)
    val inputValue = new Text("Hello")

    // Call map function
    mapper.map(inputKey, inputValue, outputCollector, reporter)

    // Verify that no embeddings were emitted
    outputCollector.collectedData shouldBe empty
  }

  "EmbeddingMapper" should "correctly process a large input and emit multiple tokens and embeddings" in {
    val mapper = new Embedding.EmbeddingMapper
    val outputCollector = new TestOutputCollector()
    val reporter = Reporter.NULL

    // Simulate a large text input
    val inputKey = new LongWritable(1)
    val inputValue = new Text("The quick brown fox jumps over the lazy dog. " * 100)

    // Call map function
    mapper.map(inputKey, inputValue, outputCollector, reporter)

    // Verify that the mapper emitted tokens
    outputCollector.collectedData should not be empty
    outputCollector.collectedData.length should be > 100 // Ensure multiple tokens were processed
  }

  "EmbeddingReducer" should "handle invalid embeddings gracefully" in {
    val reducer = new Embedding.EmbeddingReducer
    val outputCollector = new TestOutputCollector()
    val reporter = Reporter.NULL

    // Simulate invalid embeddings input
    val inputKey = new Text("test")
    val invalidEmbeddings = new TestIterator(Seq("[invalid, embedding]", "[not, a, number]"))

    // Call reduce function
    reducer.reduce(inputKey, invalidEmbeddings, outputCollector, reporter)

    // Verify that the reducer output something, likely with an error indicator
    outputCollector.collectedData should not be empty
    val (key, value) = outputCollector.collectedData.head
    key.toString shouldBe "test"
    value.toString should include("NaN") // Likely result from invalid embeddings
  }

  "embeddingMain" should "fail when the input path is invalid" in {
    val inputPath = "non/existent/path"
    val outputPath = "src/test/resources/output"

    // Call embeddingMain and expect it to fail
    val thrown = intercept[Exception] {
      Embedding.embeddingMain(inputPath, outputPath)
    }

    // Validate that the error message corresponds to an invalid path
    thrown.getMessage should include("File not found")
  }

//  "embeddingMain" should "handle empty input files without errors" in {
//    // Setup empty input and output paths
//    val inputPath = "src/test/resources/empty_input"
//    val outputPath = "src/test/resources/output"
//    Files.createDirectories(Paths.get(inputPath)) // Ensure the input directory exists
//    Files.createFile(Paths.get(s"$inputPath/empty_file.txt")) // Create an empty file
//
//    // Call embeddingMain function
//    val job = Embedding.embeddingMain(inputPath, outputPath)
//
//    // Validate that the job completed successfully
//    job.isComplete shouldBe true
//    job.isSuccessful shouldBe true
//  }

  "EmbeddingMain" should "successfully configure and run the MapReduce job" in {
    val random = Math.random()
    val job = Embedding.embeddingMain("/input", "/output/e2e_test_" + random)

    // Validate that the job completed successfully
    job.isComplete shouldBe true
    job.isSuccessful shouldBe true
  }
}
