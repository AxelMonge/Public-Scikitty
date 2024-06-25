"use client";
import SKTryDemo from "../../../components/SKComponents/SKTryDemo";
import TopicHeader from "@/app/components/Topic/TopicHeader";

const DESC = `
    A Decision Tree is a supervised learning model used for classification and regression tasks. 
    It handles categorical variables without binarizing or encoding them. You can test it using 
    the datasets we offer, which illustrate metrics and the confusion matrix. Implemented in Python, 
    this decision tree is compiled to Prolog for making predictions on the web page by simply inputting 
    the value of each feature.
`

const SciKitty = ({ }) => {
    return (
        <main>
            <article className="text-[#ffffff] pb-10">
                <TopicHeader title="SciKitty Decision Tree" description={DESC} />
                <SKTryDemo />
            </article>
        </main>
    )
}

export default SciKitty;