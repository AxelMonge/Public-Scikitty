import Image from "next/image";
import TopicForm from "../../Topic/TopicForm";

const DoQuestion = ({ values, questionTarget, fileName }) => {
    return (
        <section className="mt-5">
            <p className="mb-3">
                3. Make a prediction using the feature parameters.
            </p>
            <TopicForm values={values} questionTarget={questionTarget} fileName={fileName} />
        </section>
    )
}

export default DoQuestion;