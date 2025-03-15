#!/usr/bin/env node
const { program } = require('commander');
const inquirer = require('inquirer');
const { logger } = require('./config/config');
const TemporalCore = require('./src/temporal-core');
const ContextEngine = require('./src/context-engine');
const MultiModelRouter = require('./src/multi-model/router');

// Initialize core components
const temporalCore = new TemporalCore('./data/temporal');
const contextEngine = new ContextEngine(temporalCore, './data/context');
const multiModelRouter = new MultiModelRouter(contextEngine);

async function initialize() {
  try {
    await temporalCore.initialize();
    await contextEngine.initialize();
    logger.info('PTOLEMY system initialized successfully');
    return true;
  } catch (error) {
    logger.error(`Failed to initialize PTOLEMY system: ${error.message}`);
    return false;
  }
}

program.name('ptolemy').description('PTOLEMY AI platform').version('0.1.0');

program.command('init')
  .description('Initialize new project')
  .action(async () => {
    const initialized = await initialize();
    if (initialized) {
      const answers = await inquirer.prompt([
        { type: 'input', name: 'projectName', message: 'Project name:', default: 'my-ptolemy-project' }
      ]);
      await temporalCore.recordEvent('project_initialized', { name: answers.projectName });
      logger.info(`Project ${answers.projectName} initialized.`);
    }
  });

program.command('generate')
  .description('Generate code from a prompt')
  .argument('<prompt>', 'The prompt to generate code from')
  .option('-m, --model <model>', 'Model to use (architect, implementer, reviewer, integrator)', 'implementer')
  .action(async (prompt, options) => {
    await initialize();
    try {
      const result = await multiModelRouter.routeTask(prompt, options.model);
      console.log('\nGenerated output:');
      console.log('=================');
      console.log(result);
    } catch (error) {
      logger.error(`Generation failed: ${error.message}`);
    }
  });

program.command('chain')
  .description('Run a multi-stage prompt chain')
  .argument('<initialPrompt>', 'The initial prompt to start the chain')
  .option('-c, --config <configFile>', 'Path to chain configuration file')
  .action(async (initialPrompt, options) => {
    await initialize();
    try {
      // Default chain if no config provided
      const stages = [
        { name: 'architecture', modelType: 'architect', nextPrompt: 'Implement this architecture' },
        { name: 'implementation', modelType: 'implementer', nextPrompt: 'Review this implementation' },
        { name: 'review', modelType: 'reviewer' }
      ];
      
      const results = await multiModelRouter.routeMultiStage(initialPrompt, stages);
      
      console.log('\nChain Results:');
      console.log('==============');
      for (const result of results) {
        console.log(`\n## Stage: ${result.stage}`);
        console.log(result.output);
      }
    } catch (error) {
      logger.error(`Chain execution failed: ${error.message}`);
    }
  });

program.parse(process.argv);

// If no command is provided, show help
if (!process.argv.slice(2).length) {
  program.outputHelp();
}
