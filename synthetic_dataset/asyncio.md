asyncio
=======

Task - evaluation of each test_case is a task:

tasks.append(asyncio.create_task(task))

asyncio.gather(*tasks)

evaluate.py evaluate() ran in run_async mode.
```python
def evaluate(test_cases, metrics, ...) -> EvaluationResult:
    with capture_evaluation_run("evaluate()"):
        if async_config.run_async:
            loop = get_or_create_event_loop()
            test_results = loop.run_until_complete(
                a_execute_test_cases(
                    test_cases,
                    metrics,
                    identifier=identifier,
                    ignore_errors=error_config.ignore_errors,
                    skip_on_missing_params=error_config.skip_on_missing_params,
                    use_cache=cache_config.use_cache,
                    save_to_disk=cache_config.write_cache,
                    verbose_mode=display_config.verbose_mode,
                    show_indicator=display_config.show_indicator,
                    throttle_value=async_config.throttle_value,
                    max_concurrent=async_config.max_concurrent,
                )
            )
```


```python
async def a_execute_test_cases(
    test_cases: Union[
        List[LLMTestCase], List[ConversationalTestCase], List[MLLMTestCase]
    ],
    metrics: Union[
        List[BaseMetric],
        List[BaseConversationalMetric],
        List[BaseMultimodalMetric],
    ],
    ignore_errors: bool,
    skip_on_missing_params: bool,
    use_cache: bool,
    show_indicator: bool,
    throttle_value: int,
    max_concurrent: int,
    save_to_disk: bool = False,
    verbose_mode: Optional[bool] = None,
    identifier: Optional[str] = None,
    test_run_manager: Optional[TestRunManager] = None,
    _use_bar_indicator: bool = True,
    _is_assert_test: bool = False,
) -> List[TestResult]:

    semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_with_semaphore(func: Callable, *args, **kwargs):
        async with semaphore:
            return await func(*args, **kwargs)

    test_results: List[Union[TestResult, MLLMTestCase]] = []
    tasks = []

    for test_case in test_cases:
        # for each test case,
        # invoke `a_execute_llm_test_cases` to evaluate all metrics
        task = execute_with_semaphore(
            func=a_execute_llm_test_cases,
            metrics=copied_llm_metrics,
            test_case=test_case,
            test_run_manager=test_run_manager,
            test_results=test_results,
            count=llm_test_case_counter,
            test_run=test_run,
            ignore_errors=ignore_errors,
            skip_on_missing_params=skip_on_missing_params,
            use_cache=use_cache,
            show_indicator=show_indicator,
            _use_bar_indicator=_use_bar_indicator,
            _is_assert_test=_is_assert_test,
            progress=progress,
            pbar_id=pbar_id,
        )

        tasks.append(asyncio.create_task(task))
        await asyncio.sleep(throttle_value)

    await asyncio.gather(*tasks)

```

Eval all metrics for one test case. Ran with a semaphore.
```python
# eval all `metrics` for one `test_case`
async def a_execute_llm_test_cases(
    metrics: List[BaseMetric],
    test_case: LLMTestCase,
    test_run_manager: TestRunManager,
    test_results: List[Union[TestResult, MLLMTestCase]],
    count: int,
    test_run: TestRun,
    ignore_errors: bool,
    skip_on_missing_params: bool,
    use_cache: bool,
    show_indicator: bool,
    _use_bar_indicator: bool,
    _is_assert_test: bool,
    progress: Optional[Progress] = None,
    pbar_id: Optional[int] = None,
):
    ##### Metric Calculation #####
    # api_test_case uses:
    # api_test_case.update_metric_data(metric_data)
    # api_test_case.update_run_duration(run_duration)
    # test_run_manager.update_test_run(api_test_case, test_case)
    # test_results.append(create_test_result(api_test_case))
    api_test_case = create_api_test_case(
        test_case=test_case,
        index=count if not _is_assert_test else None
    )
    test_start_time = time.perf_counter()

    # for each metric, add task to 
    await measure_metrics_with_indicator(
        metrics=metrics,
        test_case=test_case,
        cached_test_case=cached_test_case,
        skip_on_missing_params=skip_on_missing_params,
        ignore_errors=ignore_errors,
        show_indicator=show_metrics_indicator,
        pbar_eval_id=pbar_test_case_id,
        progress=progress,
    )

    for metric in metrics:
        if metric.skipped:
            continue

        metric_data = create_metric_data(metric)
        api_test_case.update_metric_data(metric_data)

        if metric.error is None:
            cache_metric_data = deepcopy(metric_data)
            cache_metric_data.evaluation_cost = (
                0  # Create new copy and save 0 for cost
            )
            updated_cached_metric_data = CachedMetricData(
                metric_data=cache_metric_data,
                metric_configuration=Cache.create_metric_configuration(metric),
            )
            new_cached_test_case.cached_metrics_data.append(
                updated_cached_metric_data
            )

    test_end_time = time.perf_counter()
    run_duration = test_end_time - test_start_time
    # Quick hack to check if all metrics were from cache
    if run_duration < 1:
        run_duration = 0
    api_test_case.update_run_duration(run_duration)

    ### Update Test Run ###
    test_run_manager.update_test_run(api_test_case, test_case)

    ### Cache Test Run ###
    global_test_run_cache_manager.cache_test_case(
        test_case,
        new_cached_test_case,
        test_run.hyperparameters,
    )
    global_test_run_cache_manager.cache_test_case(
        test_case,
        new_cached_test_case,
        test_run.hyperparameters,
        to_temp=True,
    )

    test_results.append(create_test_result(api_test_case))
    update_pbar(progress, pbar_id)
```

Implementation of measure_metrics_with_indicator.
`tasks` queue is used for each metric.
Progress has `self._lock = RLock()`
```python

async def measure_metrics_with_indicator(
    metrics: List[
        Union[BaseMetric, BaseMultimodalMetric, BaseConversationalMetric]
    ],
    test_case: Union[LLMTestCase, MLLMTestCase, ConversationalTestCase]
...):

    with Progress(
        SpinnerColumn(style="rgb(106,0,255)"),
        BarColumn(bar_width=60),
        TextColumn("[progress.description]{task.description}"),
        transient=False,
    ) as progress:
        tasks = []
        for metric in metrics:
            # add a new task to Progress dispaly. Uses re-entrant thread lock, RLock
            task_id = progress.add_task(
                description=format_metric_description(
                    metric, async_mode=True
                ),
                total=100,
            )
            tasks.append(
                measure_metric_task(
                    task_id,
                    progress,
                    metric,
                    test_case,
                    cached_test_case,
                    ignore_errors,
                    skip_on_missing_params,
                    _in_component=_in_component,
                )
            )
        await asyncio.gather(*tasks)


async def measure_metric_task(
    task_id,
    progress,
    metric: Union[BaseMetric, BaseMultimodalMetric, BaseConversationalMetric],
    test_case: Union[LLMTestCase, MLLMTestCase, ConversationalTestCase],
    ...
):
    while not progress.finished:
        await metric.a_measure(
        test_case,
        _show_indicator=False,
        _in_component=_in_component,
        )
        finish_text = "Done"

```