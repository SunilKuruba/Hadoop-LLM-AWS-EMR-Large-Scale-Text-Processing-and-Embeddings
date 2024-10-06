import com.typesafe.config.ConfigFactory
import org.apache.hadoop.fs.Path
import org.apache.hadoop.mapred.{FileInputFormat, FileOutputFormat, JobConf}

import java.time.LocalDateTime
import java.util.UUID.randomUUID
import scala.jdk.CollectionConverters.*

/**
 * The `JobConfig` object provides configuration setup for Hadoop MapReduce jobs,
 * with support for different environments (e.g., production, local, test).
 *
 * It loads environment-specific settings from `application.conf` and configures the job's input and output paths,
 * as well as other necessary settings required for running Hadoop jobs.
 */
object JobConfig {

  /**
   * Enumeration representing the different environments in which the job can run.
   * Available environments are:
   * - `prod`: Production environment.
   * - `local`: Local development environment.
   * - `test`: Testing environment.
   */
  enum Environment:
    case prod, local, test

  /** The environment in which the job is currently running. Defaults to `local`. */
  var environment: Environment = Environment.prod

  /**
   * Creates and configures a `JobConf` object for a Hadoop MapReduce job.
   * The configuration parameters, such as job name, file system, input path, and output path,
   * are loaded from the configuration file (`application.conf`) based on the current environment.
   *
   * @param jobNamePath The configuration path in `application.conf` that contains the job name.
   * @return A fully configured `JobConf` object ready for submission to a Hadoop job.
   */
  def createJob(jobNamePath: String): JobConf = {
    val config = ConfigFactory.load
    val jobConf: JobConf = new JobConf(this.getClass)

    // Set job configuration parameters
    val jobName = config.getString(jobNamePath)
    jobConf.setJobName(config.getString(jobNamePath))

    // Set file system according to the environment
    val fileSystem = config.getString(s"hadoop.fileSystem.$environment")
    if(fileSystem.nonEmpty) jobConf.set("fs.defaultFS", fileSystem)
    jobConf.setNumReduceTasks(config.getInt(s"hadoop.numReducer"))

    // Set the maximum split size for input files
//    jobConf.setLong("mapreduce.input.fileinputformat.split.maxsize", config.getLong("hadoop.maxSplitSize"))
    jobConf.setLong("mapreduce.input.fileinputformat.split.minsize", config.getLong("hadoop.maxSplitSize"))

    // Set input and output paths based on the environment
    val inputPath = config.getString(s"io.inputdir.$environment")
    val outputPath = config.getString(s"io.outputdir.$environment") + jobName + randomUUID.toString

    // Configure input and output paths for the job
    FileInputFormat.setInputPaths(jobConf, new Path(inputPath))
    FileOutputFormat.setOutputPath(jobConf, new Path(outputPath))

    // Return the configured JobConf object
    jobConf
  }
}
